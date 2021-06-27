# orbit-est-projection-mkf
<br/>
Satellite Orbit Estimation using a projection based modified Kalman Filter. <br/>
<br/>
The standard Kalman filter is updated with a modified measurement equation which eliminates unwanted observables in the filter state by projection along a hyperplane. The resulting equation is also observable and has all convergence properties of the standard Kalman Filter. The primary advantage is the reduction of noise in the mesaurement update equation which in turn results in faster convergence. <br/>
Related work : Schmidt Kalman Filter<br/>
<br/>
Estimation Error along X,Y,Z for three satellite in geo sync orbit.<br/>
Satellite A (Red) : Filter enabled. Satellite B(green) C(blue): Filter disabled
<br/><br/>

[<img src="img/SatAEKF_dx.PNG" width="250px"/>](img/SatAEKF_dx.PNG)
[<img src="img/SatAEKF_dy.PNG" width="250px"/>](img/SatAEKF_dy.PNG)
[<img src="img/SatAEKF_dz.PNG" width="250px"/>](img/SatAEKF_dz.PNG)

Root Sum Square Error , three satellites<br/><br/>

[<img src="img/SatAEKF_rss.PNG" width="250px"/>](img/SatAEKF_rss.PNG)
[<img src="img/SatAEKF_rss2.PNG" width="250px"/>](img/SatAEKF_rss2.PNG)

<br/>

