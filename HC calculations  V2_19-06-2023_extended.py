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


def mult(a,b):
    mu=np.array([[0,0,0],[0,0,0],[0,0,0]])
    
    for i in range(0,3):
        for j in range(0,3):
            mu[i][j]=a[i][0]*b[0][j]+a[i][1]*b[1][j]+a[i][2]*b[2][j]
    print("Product of matrixes is=",mu)

#------------------------- defining theta of the vector------------------------------
def theta(vector):
    theta=180*(1/np.pi)*np.arccos(vector[2]/np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2))
    return theta

#------------------------- defining phi of the vector--------------------------------------
def phi(vector):
    phi=180*(1/np.pi)*np.arccos(vector[0]/np.sqrt(vector[0]**2+vector[1]**2))
    return phi

#--------------------------------------refraction in vector form--------------
def refract(refr,incident,refracted):
    mu=1/refr           # 1/refractive index
    norm=[-1,0,0]
    ni_2=incident[0]*refracted[0]+incident[1]*refracted[1]+incident[2]*refracted[2]
    refracted[0]=mu*incident[0]+norm[0]*np.sqrt(1-mu**2*(1-ni_2))-mu*norm[0]*np.sqrt(ni_2)
    refracted[1]=mu*incident[1]+norm[1]*np.sqrt(1-mu**2*(1-ni_2))-mu*norm[1]*np.sqrt(ni_2)
    refracted[2]=mu*incident[2]+norm[2]*np.sqrt(1-mu**2*(1-ni_2))-mu*norm[2]*np.sqrt(ni_2)

#---------------------vector module-------------------------------------------
def module(v):
    return np.sqrt(v[0]**2+v[1]**2+v[2]**2)


#-----------------test product of the vector and matrix-----------------------

def test_product(vector, angle,rotated):
    M=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]])
    M[0][0]=-1          # rotation matrix's elements
    M[0][1]=2
    M[0][2]=1
    M[1][0]=0
    M[1][1]=3
    M[1][2]=-2
    M[2][0]=1
    M[2][1]=1
    M[2][2]=2

    rotated[0]=vector[0]*M[0][0]+vector[1]*M[0][1]+vector[2]*M[0][2]
    rotated[1]=vector[0]*M[1][0]+vector[1]*M[1][1]+vector[2]*M[1][2]
    rotated[2]=vector[0]*M[2][0]+vector[1]*M[2][1]+vector[2]*M[2][2]
    
    #return rotated

#--------------------------- Determinant finding------------------------------
def det(_m):
    det=_m[0][0]*_m[1][1]*_m[2][2]+_m[1][0]*_m[2][1]*_m[0][2]+_m[0][1]*_m[1][2]*_m[2][0]-_m[2][0]*_m[1][1]*_m[0][2]-_m[1][0]*_m[0][1]*_m[2][2]-_m[0][0]*_m[2][1]*_m[1][2]
    return det

#--------------------------around Z Rotation procedure ------------------------
def around_Z_rotation(vector, angle,rotated):
    M=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]])
    M[0][0]=np.cos(angle)          # rotation matrix's elements
    M[0][1]=-np.sin(angle)
    M[0][2]=0
    M[1][0]=np.sin(angle)
    M[1][1]=np.cos(angle)
    M[1][2]=0
    M[2][0]=0
    M[2][1]=0
    M[2][2]=1

    rotated[0]=vector[0]*M[0][0]+vector[1]*M[0][1]+vector[2]*M[0][2]
    rotated[1]=vector[0]*M[1][0]+vector[1]*M[1][1]+vector[2]*M[1][2]
    rotated[2]=vector[0]*M[2][0]+vector[1]*M[2][1]+vector[2]*M[2][2]
    
    #return rotated

#-------------------------- around Y Rotation procedure -------------------------------
def around_Y_rotation(vector, angle,rotated):
    M=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]])
    
    M[0][0]=np.cos(angle)          # rotation matrix's elements
    M[0][1]=0.0
    M[0][2]=np.sin(angle)
    M[1][0]=0.0
    M[1][1]=1.0
    M[1][2]=0.0
    M[2][0]=-np.sin(angle)
    M[2][1]=0.0
    M[2][2]=np.cos(angle)

    rotated[0]=vector[0]*M[0][0]+vector[1]*M[0][1]+vector[2]*M[0][2]
    rotated[1]=vector[0]*M[1][0]+vector[1]*M[1][1]+vector[2]*M[1][2]
    rotated[2]=vector[0]*M[2][0]+vector[1]*M[2][1]+vector[2]*M[2][2]
    
    return rotated

#-------------------------- around X Rotation procedure -------------------------------
def around_X_rotation(vector, angle,rotated):
    M=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]])
    
    M[0][0]=1.0          # rotation matrix's elements
    M[0][1]=0.0
    M[0][2]=0.0
    M[1][0]=0.0
    M[1][1]=np.cos(angle)
    M[1][2]=-1.0*np.sin(angle)
    M[2][0]=0.0
    M[2][1]=np.sin(angle)
    M[2][2]=np.cos(angle)

    rotated[0]=vector[0]*M[0][0]+vector[1]*M[0][1]+vector[2]*M[0][2]
    rotated[1]=vector[0]*M[1][0]+vector[1]*M[1][1]+vector[2]*M[1][2]
    rotated[2]=vector[0]*M[2][0]+vector[1]*M[2][1]+vector[2]*M[2][2]
    
    return rotated

#--------------------refractive indice----------------------------------------
def nx(wl):
    nx=np.sqrt(3.29100+0.04140/(wl*wl-0.03978)+9.35522/(wl*wl-31.45571))
    return nx

def ny(wl):
    ny=np.sqrt(3.45018+0.04341/(wl*wl-0.04597)+16.98825/(wl*wl-39.43799))
    return ny

def nz(wl):
    nz=np.sqrt(4.59423+0.06206/(wl*wl-0.04763)+110.80672/(wl*wl-86.12171))
    return nz

def nxz(theta_,lambda_):
    nxz=nx(lambda_)*nz(lambda_)/np.sqrt(nx(lambda_)**2*np.sin(theta_)**2+nz(lambda_)**2*np.cos(theta_)**2)
    return nxz

#---------------initial parameters--------------------------------------------
ROC=600             # radius of curvature
f=ROC/2             # focal length
N=11                # number of bounces on one mirror
M=N-1              # parameter M   M=N-1 means the longest configuration
thickness=100        # nonlinear crysral thickness
theta_HC=M*np.pi/N     # angle between bounces
wavelength=1.08         # um
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
"""
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
"""

#---------Typing some calculated parameters for each one pass------------------
#"""
for _count in range(0,2*N):
    _phi=np.arctan(np.sqrt((x[_count]-x[_count+1])**2+(y[_count]-y[_count+1])**2)/d)
    print("pass#",_count,"-",_count+1,"dx=","{:.2f}".format(x[_count]-x[_count+1]),"dy=","{:.2f}".format(y[_count]-y[_count+1]),"AOI=","{:.2f}".format(_phi*180/np.pi))#,"r=","{:.3f}".format(np.sqrt((x[count+1])**2+(y[count+1])**2)))
#"""          
#---------------------------AOI VS M-------------------------------------------
#"""

for count2 in range(1,N):           # parameter M
    theta_=count2*np.pi/N
    d=ROC*(1-np.cos(theta_))
    x[0]=A*0.5*np.sqrt(d/f)
    y[0]=x[0]*np.sqrt(4*f/d-1)

    xslope=-A/np.sqrt(ROC*d/2)
    yslope=0*A/np.sqrt(ROC*d/2)

    x[1]=x[0]*np.cos(theta_)+np.sqrt(d/(4*f-d))*(x[0]+2*f*xslope)*np.sin(theta_)
    y[1]=y[0]*np.cos(theta_)+np.sqrt(d/(4*f-d))*(y[0]+2*f*yslope)*np.sin(theta_)
        
    phi_=np.arctan(np.sqrt((x[0]-x[1])**2+(y[0]-y[1])**2)/d)
    print("M=",count2,"d=","{:.1f}".format(ROC*(1-np.cos((count2)*np.pi/N))),"mm, AOI=","{:.2f}".format(phi_*180/np.pi),"deg, dl=","{:.1f}".format(5000/np.cos(phi_/1.45)-5000),"um")

#"""

#--------------------wave vectors coordinates vs the pass number---------------

k=[[0, 0, 0] for _ in range(2*N)]   # wave numbers array declaration
for count in range (2*N):                                   # k vektor projections calculation
    k[count]=[x[count+1]-x[count],y[count+1]-y[count],d*(-1)**(count)] # defining x,y,z coordinates of the k vector 
    k_module=np.sqrt(k[count][0]**2+k[count][1]**2+k[count][2]**2) # k module calculation

    k[count][0]=k[count][0]/k_module        # x projection normalization
    k[count][1]=k[count][1]/k_module        # y projection normalization
    k[count][2]=k[count][2]/k_module        # z projection normalization
    print("k[",count,"]=[","{:.5f}".format(k[count][0]),"{:.5f}".format(k[count][1]),"{:.5f}".format(k[count][2]),"]","module=","{:.1f}".format(module(k[count])))

thetaPM=86.907
phiPM=0
k_rot_tmp=[[0, 0, 0] for _ in range(2*N)]   # wave numbers array declaration
k_rot=[[0, 0, 0] for _ in range(2*N)]   # wave numbers array declaration
Esh=0  
dEsh=0      # 

for count in range (2*N):
    around_X_rotation(k[count],1*np.pi/2,k_rot_tmp[count])
    
    around_Z_rotation(k_rot_tmp[count],1*np.pi/2,k[count])
    
    around_Y_rotation(k[count],-1*(np.pi/2-thetaPM*np.pi/180),k_rot[count])
    
    lc=wavelength*0.5/(ny(wavelength)+nxz(theta(k_rot[count])*np.pi/180,wavelength)-2*ny(wavelength/2))
    dEsh=np.sin(np.pi*thickness/lc)
    Esh+=dEsh
    
    print("k[",count,"]=","theta=[","{:.5f}".format(theta(k_rot[count])),"], phi=[","{:.5f}".format(phi(k_rot[count])),"]","d_eff=","{:.5f}".format(3.9*np.sin((np.pi/180)*theta(k_rot[count]))),"dEsh=","{:.5f}".format(dEsh))

lc=wavelength/(ny(wavelength)+nxz(thetaPM*np.pi/180,wavelength)-2*ny(wavelength/2))
print("no@1080=",ny(wavelength),"ne@1080=",nxz(thetaPM*np.pi/180,wavelength),"no@540=",ny(wavelength/2),"lc[um]=",lc)
print("accumulated SH electric field [a.u.]=",Esh)