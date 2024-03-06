# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 20:02:18 2023

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:42:07 2022

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt






#---------------initial parameters--------------------------------------------
ROC=600             # radius of curvature
f=ROC/2             # focal length
N=5                 # number of bounces on one mirror
M=N-1                # parameter M   M=N-1 means the longest configuration
theta_HC=M*np.pi/N     # angle between bounces
x=np.empty(2*N+1, dtype=object)     # x array creation
y=np.empty(2*N+1, dtype=object)     # y array creation
for count in range(2*N+1):          # x and y initial values
    x[count]=0
    y[count]=0
d=ROC*(1-np.cos(theta_HC))             # distance between mirrors for the certain M
A=10                                # Pattern's radius

x[0]=A*0.5*np.sqrt(d/f)             # input beam x coordinate
y[0]=x[0]*np.sqrt(4*f/d-1)          # input beam y coordinate

kx=-1                               # sign of the x slope
ky=0                                # sign of the y slope

xslope=kx*A/np.sqrt(ROC*d/2)        # x slope calculation
yslope=ky*A/np.sqrt(ROC*d/2)        # y slope calculation

for count in range (2*N):           # whole the X and Y coordinates array calculations
    x[count+1]=x[0]*np.cos((count+1)*theta_HC)+np.sqrt(d/(4*f-d))*(x[0]+2*f*xslope)*np.sin((count+1)*theta_HC)
    y[count+1]=y[0]*np.cos((count+1)*theta_HC)+np.sqrt(d/(4*f-d))*(y[0]+2*f*yslope)*np.sin((count+1)*theta_HC)
  

#-----------------------------------Plotting the patterns----------------------

fig1=plt.figure(figsize=(8,8))

plt.xlim([-15, 15])
plt.ylim([-15, 15])
fig2=plt.figure(figsize=(8,8))
plt.xlim([-15, 15])
plt.ylim([-15, 15])
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

for count in range(2*N+1):
    #print(count,"=",x[count]," ",y[count])
    if (count/2)==round(count/2):
        ax1.plot(x[count], y[count], 'or')
        ax1.annotate(str(count), (x[count],y[count]),fontsize=20) 
    else: 
        ax2.plot(x[count], y[count], 'ob')
        ax2.annotate(str(count), (x[count],y[count]),fontsize=20)


#---------Typing some calculated parameters for each one pass------------------
        
#---------------------------AOI VS M-------------------------------------------


#--------------------wave vektors coordinates vs the pass number---------------

