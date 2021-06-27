import sys
from datetime import datetime,timedelta,timezone

import numpy as np

from scipy.linalg import norm
from scipy.linalg import block_diag
from scipy.interpolate import splrep,splev

from jplephem.spk import SPK as spk

#
# Constants
#

dict_constants = {
	'earth_GM': 3.986004418e5,		#km3s-2
	'earth_R' : 6378.137,			#km

	'sun_GM'  : 1.32712440018e11,	#km3s-2
	'moon_GM' : 4.9048695e3,		#km3s-2

	'AU'	  :	149597870.,			#km
	'P0'	  : 4.56e-3	,			#kgkm-1s-2
};

#
# Earth Orientation Parameters
#

#Load EOP spline data, inertial to fixed matrix
def loadeop():

	eopspl,f2i,i2f = [],None,None

	eopd = np.loadtxt('./data/eop.txt',usecols=range(7,16)).T
	eopt = [t  for t in range(0,(7200*120)+120,120)]

	for i in range(9):
		eopspl.append(splrep(eopt,eopd[i]))
	#end for

	i2f = lambda t,spl : np.array([splev(t,spl[i]) for i in range(9)]).reshape(3,3).T  

	return eopspl,i2f
#end loadeop


#
# Solar Radiation Pressure
#

def accsrp(t,pos,solsys,dict_info,comp_ag):

	P0 = dict_info['P0']   			# Solar radiation pressure at 1 AU in kgkm-1s-2
	AU = dict_info['AU'] 			# 1 AU in km
	Re = dict_info['earth_R']		# Radius of earth  in km
	A  = dict_info['srp_area']		# Area of satellite in km2
	m  = dict_info['srp_mass']		# Mass of satellite in kg
	Cr = dict_info['srp_cr']   		# Coefficent of reflectivity (force parameter)
	rs = solsys['sun']				# Sun position in km

	rse = rs / norm(rs)		#sun unit vector
	sunproj = np.dot(pos,rse) #projection of satellite along sun vector  

	r = (pos - rs)			#spacecraft w.r.t sun 
	r3 = norm(r)**3.		#cube of distance 

	#Compute illumination 
	illum = 1.0 if ((sunproj > 0.) or (norm(pos-(sunproj*rse)) > Re)) else 0.0

	#Compute acceleration paritial w.r.t Cr
	apart = ((A*P0*(AU**2))/(m*r3))*r

	#Compute acceleration (ms-2)
	acc = illum * Cr * apart

	return acc,np.zeros((3,3)),apart


#
# Third Body Attraction
#

def getsolsys(t,reft,de430):

	getjd = lambda time : (time.replace(tzinfo=timezone.utc).timestamp() / 86400.0) + 2440587.5
	t = getjd(reft+timedelta(seconds=t)) 

	#compute eath position from solar system barycenter (SSB)
	ssb_earth = de430[0,3].compute(t) #SSB->EarthB
	ssb_earth += de430[3,399].compute(t) #EarthB->Earth

	#compute sun position
	sunpos = de430[0,10].compute(t) #SSB->Sun
	sunpos -= ssb_earth

	#compute moon position
	moonpos = de430[3,301].compute(t) #EB->Moon
	moonpos -= de430[3,399].compute(t) #EB->Earth

	#compute jupiter position
	juppos = de430[0,5].compute(t) #SSB->JupiterB
	juppos -= ssb_earth

	#compute venus position
	venpos = de430[0,2].compute(t) #SSB->VenusB
	venpos += de430[2,299].compute(t) #VenusB->Venus
	venpos -= ssb_earth	

	return {'sun':sunpos,'mon':moonpos,'jup':juppos,'ven':venpos}
#end getsunmoon

def accpointmass(r,s,MU):

	pos = r - s
	pmag = norm(pos)
	pmag2 = pmag*pmag
	pmag5 = pmag2*pmag2*pmag

	acc = -MU         * ( (pos/(pmag2*pmag)) + (s/(norm(s)**3.)     ) )

	ag  = -(MU/pmag5) * ( (pmag2*np.eye(3) ) - (3.*np.outer(pos,pos)) )

	return acc,ag
#end

def accthirdbody(t,pos,dict_info,comp_ag):

	#get solar system bodies at time t
	solsys = getsolsys(t,dict_info['body_reft'],dict_info['body_de430'])

	tbacc,tbag = np.zeros(3),np.zeros((3,3))

	#get acceleration due to sun
	acc,ag = accpointmass(pos,solsys['sun'],dict_info['sun_GM'])
	tbacc += acc; tbag += ag

	#get acceleration due to sun
	acc,ag = accpointmass(pos,solsys['mon'],dict_info['moon_GM'])
	tbacc += acc; tbag += ag

	return tbacc,tbag,solsys
#end


#
# Geo Potential
#

#Load geo coefficents
def loadgeocoeff(norm=True):
	C,S = {},{}

	C[0],S[0] = {},{}
	C[0][0],S[0][0] = 1.0,0.0
	C[1],S[1] = {},{}
	C[1][0],S[1][0] = 0.0,0.0
	C[1][1],S[1][1] = 0.0,0.0

	with open("./data/egm96.txt") as f:
		for line in f:
			parts = line.split()
			n,m = int(parts[0]),int(parts[1])
			cnm,snm = float(parts[2]),float(parts[3])
			if(n not in C):
				C[n] = {}; S[n] = {};
			#end if
			C[n][m] = cnm; S[n][m] = snm
		#end for
	#end with
	
	if(norm):
		return C,S
	#end if

	#denormalize geo coefficents (upto n = 50)
	with open("./data/denorm.txt") as f:
		for line in f:
			parts = line.split()
			n,m,denorm = int(parts[0]),int(parts[1]),float(parts[2])
			C[n][m] *= denorm
			S[n][m] *= denorm
		#end for
	#end with

	return C,S
#end loadgeocoeff

#Compute acceleration due to geopotential
def accgeo(t,pos,dict_info,comp_ag=True):

	try:
		#constants
		MU,R = dict_info['earth_GM'],dict_info['earth_R']
		#eop
		eopspl,i2f = dict_info['eop_spl'],dict_info['eop_i2f']
		#geo
		C,S = dict_info['geo_C'],dict_info['geo_S']
		nmax,mmax = dict_info['geo_n'],dict_info['geo_m']
	except:
		print("accgeo: insufficent information")
		sys.exit(-1)
	#end try
	

	#transformation matrix from inertial to fixed
	trans = i2f(t,eopspl)

	V,W = {},{}

	#eci to ecf
	pos = trans @ pos

	rmag2 = np.dot(pos,pos)
	rho = (R*R) / rmag2
	r0 = (R/rmag2)*pos

	#Compute V,W recursively
	V[0],W[0] = {},{}; V[1],W[1] = {},{}

	V[0][0],W[0][0] = R/np.sqrt(rmag2) , 0.0
	V[1][0],W[1][0] = r0[2]*V[0][0] , 0.0

	for n in range(2,nmax+2+1):
		V[n],W[n] = {},{}
		V[n][0] = ((2*n-1) * r0[2] * V[n-1][0] - (n-1) * rho * V[n-2][0]) / n
		W[n][0] = 0.0
	#end for

	for m in range(1,mmax+2+1):
		V[m][m] = (2*m-1) * ( r0[0]*V[m-1][m-1] - r0[1]*W[m-1][m-1] )
		W[m][m] = (2*m-1) * ( r0[0]*W[m-1][m-1] + r0[1]*V[m-1][m-1] )

		if(m <= nmax+1):
			V[m+1][m] = (2*m+1) * r0[2] * V[m][m]
			W[m+1][m] = (2*m+1) * r0[2] * W[m][m]
		#end if

		for n in range(m+2,nmax+2+1):
			V[n][m] = ( (2*n-1)*r0[2]*V[n-1][m] - (n+m-1)*rho*V[n-2][m] ) / (n-m)
			W[n][m] = ( (2*n-1)*r0[2]*W[n-1][m] - (n+m-1)*rho*W[n-2][m] ) / (n-m)
		#end for
	#end for

	#compute accelerations
	acc = np.zeros(3)
	for m in range(mmax+1):
		for n in range(m,nmax+1):
			if(m == 0):
				acc -= C[n][0] * np.array([V[n+1][1],W[n+1][1],(n+1)*V[n+1][0]])
			else:
				f = 0.5 * (n-m+1) * (n-m+2)
				acc[0] +=	+ 0.5 * ( - C[n][m] * V[n+1][m+1] - S[n][m] * W[n+1][m+1] ) \
							+   f * ( + C[n][m] * V[n+1][m-1] + S[n][m] * W[n+1][m-1] )
				acc[1] +=   + 0.5 * ( - C[n][m] * W[n+1][m+1] + S[n][m] * V[n+1][m+1] ) \
							+   f * ( - C[n][m] * W[n+1][m-1] + S[n][m] * V[n+1][m-1] )
				acc[2] += (n-m+1) * ( - C[n][m] * V[n+1][m]   - S[n][m] * W[n+1][m] )
			#end if-else
		#end for
	#end for

	acc *= MU / (R*R)

	#ecf to eci
	acc = trans.T @ acc

	if(comp_ag == False):
		return acc,np.zeros((3,3))
	#end if

	#compute acceleration gradient
	ag = np.zeros((3,3))
	for m in range(mmax+1):
		for n in range(m,nmax+1):
			f  = (n-m+1) * (n-m+2)
			f3 = f  * (n-m+3)
			f4 = f3 * (n-m+4) 
			if(m == 0):

				agx = 0.5 * ( (+C[n][0]*V[n+2][2]) -  f * (+C[n][0]*V[n+2][0]) )
				agz =   f * ( (+C[n][0]*V[n+2][0]) ) 
				ag[0,0] += agx
				ag[2,2] += agz
				ag[1,1] -= agx + agz

				tmp = 0.5   * ( + C[n][0]*W[n+2][2] )
				ag[0,1] += tmp
				ag[1,0] += tmp

				tmp = (n+1) * ( + C[n][0]*V[n+2][1] )
				ag[0,2] += tmp
				ag[2,0] += tmp

				tmp = (n+1) * ( + C[n][0]*W[n+2][1] )
				ag[1,2] += tmp
				ag[2,1] += tmp

			elif(m == 1):

				agx = 0.25 * ( (+C[n][1]*V[n+2][3]+S[n][1]*W[n+2][3]) + f * (-3*C[n][1]*V[n+2][1]-S[n][1]*W[n+2][1]) )
				agz =    f * ( (+C[n][1]*V[n+2][1]+S[n][1]*W[n+2][1]) )
				ag[0,0] += agx
				ag[2,2] += agz
				ag[1,1] -= agx + agz

				tmp = 0.25 * ( (+C[n][1]*W[n+2][3]-S[n][1]*V[n+2][3]) + f * (  -C[n][1]*W[n+2][1]-S[n][1]*V[n+2][1]) )
				ag[0,1] += tmp
				ag[1,0] += tmp

				tmp = 0.5 * ( n * (+C[n][1]*V[n+2][2]+S[n][1]*W[n+2][2]) +f3*( -C[n][1]*V[n+2][0]-S[n][1]*W[n+2][0]) )
				ag[0,2] += tmp
				ag[2,0] += tmp

				tmp = 0.5 * ( n * (+C[n][1]*W[n+2][2]-S[n][1]*V[n+2][2]) +f3*( -C[n][1]*W[n+2][0]-S[n][1]*V[n+2][0]))
				ag[1,2] += tmp
				ag[2,1] += tmp 
			else:

				agx = 0.25 * ( (+C[n][m]*V[n+2][m+2]+S[n][m]*W[n+2][m+2]) + 2*f*(-C[n][m]*V[n+2][m]-S[n][m]*W[n+2][m]) + f4*(+C[n][m]*V[n+2][m-2]+S[n][m]*W[n+2][m-2]) )
				agz =    f * ( (+C[n][m]*V[n+2][m]+S[n][m]*W[n+2][m]) )
				ag[0,0] += agx
				ag[2,2] += agz
				ag[1,1] -= agx + agz

				tmp = 0.25 * ( (+C[n][m]*W[n+2][m+2]-S[n][m]*V[n+2][m+2]) +  f4*(-C[n][m]*W[n+2][m-2]-S[n][m]*V[n+2][m-2]) )
				ag[0,1] += tmp
				ag[1,0] += tmp

				tmp = 0.5 * ( n * (+C[n][m]*V[n+2][m+1]+S[n][m]*W[n+2][m+1]) +f3*(-C[n][m]*V[n+2][m-1]-S[n][m]*W[n+2][m-1]) )
				ag[0,2] += tmp
				ag[2,0] += tmp

				tmp = 0.5 * ( n * (+C[n][m]*W[n+2][m+1]-S[n][m]*V[n+2][m+1]) +f3*( -C[n][m]*W[n+2][m-1]-S[n][m]*V[n+2][m-1]))
				ag[1,2] += tmp
				ag[2,1] += tmp

			#end if-elif-else
		#end for
	#end for

	ag *= MU / (R*R*R)

	#ecf to eci
	ag = trans.T @ ag @ trans

	return acc,ag
#end accgeo


#
# acceleration function
#
def accfun(t,pos,dict_info,comp_ag):

	finacc,finag = np.zeros(3),np.zeros((3,3))

	#acceleration due to geo
	acc,ag = accgeo(t,pos,dict_info,comp_ag)
	finacc += acc; finag += ag

	#acceleration due to third body
	acc,ag,solsys = accthirdbody(t,pos,dict_info,comp_ag)
	finacc += acc; finag += ag

	#acceleration due to srp
	acc,ag,asn = accsrp(t,pos,solsys,dict_info,comp_ag)
	finacc += acc; finag += ag
	
	return finacc,finag,asn
#end accfun

#
# delta y function for integration (sv and st)
#
def dysvstfun(t,y,dict_info):

	neq = len(y) // 2
	p = dict_info['p']
	
	#Compute acceleration and accelration gradient

	pos = y[:3]	#First 3 elements for x,y,z
	acc,ag,asn = accfun(t,pos,dict_info,True)
	
	#Compute the block diagonal
	A = block_diag(ag,ag,ag,ag,ag,ag)
	for _ in range(p):
		A = block_diag(A,ag)
	#end for

	return np.hstack([y[neq:],acc,np.dot(A,y[3:neq])]) + np.hstack([np.zeros(2*neq-3*p),asn])
#end dyfunsvst
