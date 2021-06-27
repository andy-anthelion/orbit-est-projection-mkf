import sys

import numpy as np

import sys,os,math,time

from datetime import datetime,timedelta,timezone

import numpy as np
from numpy.random import uniform,triangular
from numpy.linalg import matrix_rank

from scipy.linalg import block_diag
from scipy.linalg import norm
from scipy.linalg import lstsq
from scipy.linalg import svdvals
from scipy.linalg import svd
from scipy.linalg import lu
from scipy.linalg import inv

from scipy.optimize import least_squares

from scipy.interpolate import splrep,splev

from scipy.integrate import ode

from odlib import dict_constants as DICT_CONST
from odlib import loadeop
from odlib import loadgeocoeff
from odlib import dysvstfun as allaccfun

from jplephem.spk import SPK as spk

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
import matplotlib.colorbar as cbar
from mpl_toolkits.mplot3d import Axes3D

#
# Measurement
# 

def genprojector(m,n):

	P1 = (1.-n)*np.ones((m,m)) + m*(n-1)*np.eye(m)
	P2 = np.ones((m,m)) - m*np.eye(m)

	P = np.bmat([
		[P1,P2,P2],
		[P2,P1,P2],
		[P2,P2,P1],
	])

	return P/(m*n)
#end gen projector

def meas(init,data,R):
	rho = data[:,3]
	stn = data[:,:3]
	sat = np.vstack([
		np.repeat([init[0:3]],8,axis=0),
		np.repeat([init[3:6]],8,axis=0),
		np.repeat([init[6:9]],8,axis=0),
	])

	r = norm(sat-stn,axis=1)
	return np.asarray(R@(rho-r))[0]
#end meas

def measprime(init,data,R):
	rho = data[:,3]
	stn = data[:,:3]
	sat = np.vstack([
		np.repeat([init[0:3]],8,axis=0),
		np.repeat([init[3:6]],8,axis=0),
		np.repeat([init[6:9]],8,axis=0),
	])

	r = norm(sat-stn,axis=1)
	unit = -1.*(sat-stn) / np.array([r]).T

	A = np.bmat([
		[unit[:8],np.zeros((8,6))],
		[np.zeros((8,3)),unit[8:16],np.zeros((8,3))],
		[np.zeros((8,6)),unit[16:]],
	])

	return R@A
#end measprime

# Measurement Estimation
def measest(init,orgd,E,true_pos,i,sample,sigma,max_obsv):
	#Sample each measurement atleast 5 times
	while(True):
		netpos,netposvar = np.zeros(9),np.zeros(9)
		for _ in range(sample):
			noise = sigma*triangular(-1.,0,1.,max_obsv)
			d = np.copy(orgd)
			d[:,3] += noise
			W = np.diag(np.abs(.95/noise))

			res = least_squares(meas,init,jac=measprime,args=(d,W@E),method='lm',ftol=1e-8, xtol=1e-8)	
			R = measprime(res['x'],d,W@E)

			#accumlate state and covariance 
			netpos += res['x']
			netposvar += np.diagonal(inv(R.T@R))
		#end for
		netpos = (1./sample) * netpos
		netposvar = (1./sample**2) * netposvar
		neterr = norm(true_pos[i].reshape(3,3)-netpos.reshape(3,3),axis=1)
		if(np.all(neterr < 20.)):
			break
		#end if
	#end while

	return netpos,netposvar,neterr
#end measest

#
# Model 
#

#Load orbit information
def loadodinfo():
	dict_info = {}

	#load force parameter count
	dict_info['p'] = 1

	#load constants
	dict_info['earth_GM'] = DICT_CONST['earth_GM']
	dict_info['sun_GM'] = DICT_CONST['sun_GM']
	dict_info['moon_GM'] = DICT_CONST['moon_GM']
	dict_info['earth_R'] = DICT_CONST['earth_R']
	dict_info['AU'] = DICT_CONST['AU']
	dict_info['P0'] = DICT_CONST['P0']

	#load eop
	dict_info['eop_spl'],dict_info['eop_i2f'] = loadeop()

	#load geo
	dict_info['geo_C'],dict_info['geo_S'] = loadgeocoeff(False)
	dict_info['geo_n'],dict_info['geo_m'] = 20,20

	#load third body
	dict_info['body_reft'] = datetime(2011,4,10,0,1,6,184000)
	dict_info['body_de430'] = spk.open('../expt15/data/de430.bsp')

	#load srp
	dict_info['srp_mass'] = 8.5e2
	dict_info['srp_area'] = 2.25e-6
	dict_info['srp_cr'] = 1.0

	return dict_info
#end loadodinfo

#Unload orbit information
def unloadodinfo(dict_info):

	dict_info['body_de430'].close()

	return
#end unloadodinfo

#Convert elements to sv
def ele2sv(mu,ele):
	sv = np.zeros(6) #(rx ry rz vx vy vz)
	sma = ele[0]; ecc = ele[1]; inc = ele[2]
	asc = ele[3]; arg = ele[4]; mea = ele[5]
	mach_eps = np.finfo(np.float32).eps

	#solve eccentric anomaly (E) iteratively
	mea = np.mod(mea,2.0*np.pi) 
	E = mea if ecc < 0.8 else np.pi 	
	for _ in range(15):
		f = E - ecc*np.sin(E) - mea
		E = E - (f / (1.0 - ecc*np.cos(E)))
		if(np.abs(f) < mach_eps):
			break
	#end for

	cE = np.cos(E); sE = np.sin(E)
	f = np.sqrt((1.-ecc)*(1.+ecc))

	R = sma*(1. - ecc*cE)	#distance
	V = np.sqrt(mu*sma)/R 	#speed

	sv = np.array([
		sma*(cE-ecc), sma*f*sE, 0.,
		-V*sE, V*f*cE, 0.
	])

	#Transformation to reference system
	rot_z = lambda t: np.array([[np.cos(t),np.sin(t),0.],[-np.sin(t),np.cos(t),0.],[0.,0.,1.]])
	rot_x = lambda t: np.array([[1.,0.,0.],[0.,np.cos(t),np.sin(t)],[0.,-np.sin(t),np.cos(t)]])
	PQW = rot_z(-asc) @ rot_x(-inc) @ rot_z(-arg)

	r = PQW@sv[:3]; v = PQW@sv[3:]

	return np.hstack([r,v])
#end ele2sv

#Convert sv,st,sn to y
def svstsn2y(sv,st,sn):
	y = np.zeros(sv.size + st.size + sn.size)
	l1,l2 = 0,len(y)//2
	p = sn.size // 6

	#insert sv
	y[l1:l1+3],y[l2:l2+3] = sv[:3],sv[3:]
	l1 += 3; l2 += 3

	#insert state transition
	for i in range(6):
		y[l1:l1+3],y[l2:l2+3] = st[:3,i],st[3:,i]
		l1 += 3; l2 += 3
	#end for

	#insert sensitivity
	for i in range(p):
		y[l1:l1+3],y[l2:l2+3] = sn[:3,i],sn[3:,i]
		l1 += 3; l2 += 3
	#end for

	return y
#end svstsn2y

#Convert y to svstsn
def y2svstsn(y):
	l1,l2 = 0,len(y)//2 
	p = (l2 - 21) // 3 # all apart from pos,st_pos
	sv,st,sn = np.zeros(6),np.zeros((6,6)),np.zeros((6,p))

	#remove sv
	sv[:3],sv[3:] = y[l1:l1+3],y[l2:l2+3]
	l1 += 3; l2 += 3

	#remove state transition
	for i in range(6):
		st[:3,i],st[3:,i] = y[l1:l1+3],y[l2:l2+3] 
		l1 += 3; l2 += 3
	#end for

	#remove sensitivity
	for i in range(p):
		sn[:3,i],sn[3:,i] = y[l1:l1+3],y[l2:l2+3] 
		l1 += 3; l2 += 3
	#end for

	return sv,st,sn
#end y2svstsn

def modelprop(dyfun,y0,dict_info,tt,dt,atol=1e-6,rtol=1e-9):
	y,tb,te = [y0],tt[0],tt[-1]

	if (dt > 0. and (tb - te) > 0.) or (dt < 0. and (tb - te) < 0.):
		print("Error! Check input times")
		return []
	#end if

	res = ode(dyfun).set_integrator('vode',method='adams',atol=atol,rtol=rtol)
	res.set_initial_value(y0,tb).set_f_params(dict_info)

	#print("Propagate SV ... ")
	while res.successful() and (abs(res.t - te) > 0):
		y.append(res.integrate(res.t + dt))
	#end while

	return np.vstack(y)
#end modelprop


#
# Main
#
def main(argv):

	#check which satellite to update
	try:
		satnum = int(argv[1])
	except:
		print("Error! Usage "+argv[0]+" <sat-id>(1-3)")
		sys.exit(0)
	#end try

	s1,s2 = (satnum-1)*3,satnum*3

	#load data
	print("Loading A ... ")
	data_a = np.loadtxt("../expt9/rng/rng_a.txt")
	print("Loading B ... ")
	data_b = np.loadtxt("../expt9/rng/rng_b.txt")
	print("Loading C ... ")
	data_c = np.loadtxt("../expt9/rng/rng_c.txt")

	#measurement parameters
	max_dur = 17280; max_intv = 60; max_obsv = 24
	sigma = 5e-4 # 50 cms noise
	true_pos = np.hstack([data_a[:,1:4],data_b[:,1:4],data_c[:,1:4]])
	sample = 7
	err = []; pos = []; posdev = [];

	#model parameters
	dict_info = loadodinfo()
	psn = 1e4*np.ones((6,1))

	#initial satellite orbit (t0) (satellite A)
	sv = ele2sv(dict_info['earth_GM'],
	np.array([42163.0,0.001,
		np.deg2rad(0.05),np.deg2rad(268),
		np.deg2rad(3),np.deg2rad(6)]
	))
	Psv = np.diag(np.array([1e3,1e3,1e3,3.,3.,3.])**2)
	
	dsv = np.zeros(6)
	P = np.diag(np.array([1e3,1e3,1e3,3.,3.,3.])**2)

	#projector matrix
	E = genprojector(8,3)
	#measurement matrix
	H = np.hstack([np.eye(3),np.zeros((3,3))])

	print("Processing...")
	for i in range(0,max_dur,max_intv):
		#for i in tqdm(range(0,max_dur,max_intv),ascii=True):
		#for i in range(0,max_dur,max_intv):
		ti = 5.*i
		orgd = np.vstack([data_a[i,4:].reshape((8,4)),data_b[i,4:].reshape((8,4)),data_c[i,4:].reshape((8,4))])

		#
		# Measurement update
		#

		#initial values for measurement processing
		init = np.zeros(9)
		init = true_pos[i] + 1e3*(np.ones(9) + 0.5*uniform(1.,2.,9))
		#update only sleected satellite from sv
		init[s1:s2] = sv[:3]
		
		#estimate position from measurement
		netpos,netposvar,neterr = measest(init,orgd,E,true_pos,i,sample,sigma,max_obsv)

		#get measurement and variance for satellite A
		dsvmes = netpos[s1:s2] - sv[:3]
		R = np.diag(netposvar[s1:s2]) 
		
		#compute gain
		K = P @ H.T @ inv( R + H@P@H.T )
		#update sv
		dsv += K @ (dsvmes - H@dsv)
		#update P
		P = ((np.eye(6) - K@H) @ P @ (np.eye(6) - K@H).T) + (K@R@K.T)

		# print("P(meas): ",svdvals(P))
		# print("dsv(meas):",dsv)

		#update sv
		Wsum = inv(P + Psv)
		sv += np.dot(Psv @ Wsum,dsv)
		dsv = np.dot(P @ Wsum,dsv)
		Psv = (i/(i+1))*Psv + (1./(i+1))*P

		#store values
		netpos[s1:s2] = sv[:3]
		netposvar[s1:s2] = np.diag(Psv[:3,:3])
		kferr = norm(sv[:3] - true_pos[i,s1:s2])
		neterr[satnum-1] = kferr if kferr < 10. else 10.

		pos.append(netpos)
		posdev.append(3.*np.sqrt(netposvar))
		err.append(neterr)

		#
		# Model Update
		#
		propt = np.arange(ti,ti+(5.*max_intv)+1.)
		ymod0 = svstsn2y(sv,np.eye(6),np.zeros((6*dict_info['p'],1)))
		propy = modelprop(allaccfun,ymod0,dict_info,propt,5.)

		sv,pst,psn = y2svstsn(propy[-1])
		dsv = np.dot(pst,dsv)
		P = pst @ P @ pst.T

		# print("P(model): ",svdvals(P))
		# print("dsv(model):",dsv)
	#end for
	print("Done")
	err = np.vstack(err)
	pos = np.vstack(pos)
	posdev = np.vstack(posdev)

	max_err,min_err = 10.,1.
	cnm = col.Normalize(vmax=1.1*max_err,vmin=.9*min_err)
	smap = cm.ScalarMappable(norm=cnm,cmap='jet')
	xd = range(0,max_dur,max_intv)

	def plot_pos_err(s,c):
		fig2 = plt.figure()
		ax = fig2.add_subplot(311)
		ax.plot(xd,(true_pos[0:max_dur:max_intv,s]-pos[:,s]),color='r',alpha=0.4,label='$\Delta$'+c+' (A)')
		ax.fill_between(xd,np.zeros(len(xd))-posdev[:,s],np.zeros(len(xd))+posdev[:,s],edgecolor=None,facecolor='r',alpha=0.1)
		ax.set_ylim(-1.,1.)
		ax.grid(True); ax.legend()
		
		ax = fig2.add_subplot(312)
		ax.plot(xd,(true_pos[0:max_dur:max_intv,s+3]-pos[:,s+3]),color='g',alpha=0.4,label='$\Delta$'+c+' (B)')
		ax.fill_between(xd,np.zeros(len(xd))-posdev[:,s+3],np.zeros(len(xd))+posdev[:,s+3],edgecolor=None,facecolor='g',alpha=0.1)
		ax.set_ylim(-1.,1.)
		ax.grid(True); ax.legend()
		
		ax = fig2.add_subplot(313)
		ax.plot(xd,(true_pos[0:max_dur:max_intv,s+6]-pos[:,s+6]),color='b',alpha=0.4,label='$\Delta$'+c+' (C)')
		ax.fill_between(xd,np.zeros(len(xd))-posdev[:,s+6],np.zeros(len(xd))+posdev[:,s+6],edgecolor=None,facecolor='b',alpha=0.1)
		ax.set_ylim(-1.,1.)
		ax.grid(True); ax.legend()
	#end plot_pos_err

	plot_pos_err(0,'x')
	plot_pos_err(1,'y')
	plot_pos_err(2,'z')

	fig1 = plt.figure()
	plt.plot(xd,err[:,0],'-',color='r',alpha=0.4,label='A')
	plt.plot(xd,err[:,1],'-',color='g',alpha=0.4,label='B')
	plt.plot(xd,err[:,2],'-',color='b',alpha=0.4,label='C')
	plt.grid(True)
	plt.legend()

	fig = plt.figure()
	ax = fig.add_axes([.01,.1,.9,.85],projection='3d')
	cax = fig.add_axes([.01,.05,.9,.025])
	cb1 = cbar.ColorbarBase(cax,cmap='jet',norm=cnm,orientation='horizontal')
	
	for i,j in enumerate(range(0,max_dur,max_intv)):
		abc = []
		for k in range(9):
			abc.append([true_pos[j,k],true_pos[j,k] + 500.*(true_pos[j,k]-pos[i,k])])
		#end for k
		pcol = []
		for k in range(3):
			pcol.append(col.to_hex(smap.to_rgba(err[i,k])))
		#end for k
		ax.plot(abc[0],abc[1],abc[2],alpha=.4,color=pcol[0])
		ax.plot(abc[3],abc[4],abc[5],alpha=.4,color=pcol[1])
		ax.plot(abc[6],abc[7],abc[8],alpha=.4,color=pcol[2])
	#end for ij
	
	fig2 = plt.figure()
	cs = ['r','g','b']; ls = ['A','B','C']
	for i in range(3):
		w = np.ones_like(err[:,i]) / len(err[:,i])
		plt.hist(err[:,i],bins=100,weights=w,color=cs[i],alpha=0.5, normed=False,label=ls[i])
	#endfor
	plt.grid(True)
	
	plt.show()
	return
#end main


if __name__ == '__main__':
	np.random.seed(math.floor(time.time()))
	np.set_printoptions(precision=2,suppress=False,linewidth=200)
	main(sys.argv)
	sys.exit(0)
#end if
