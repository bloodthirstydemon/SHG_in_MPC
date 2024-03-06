# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:31:24 2023

@author: akbar
"""
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants as const



################################################################################################################
'parametrs'
################################################################################################################


'''for qrartz d11 = 0.3 (pm/V) d14 = 0.008 (pm/V)
in imperical units d11 = 0.3e-12 (m/V) d14 = 0.008e-12 (m/V)'''

C = const.c
wl = 1064e-9 #(1050nm)
n1 = 1.5342
n2 = 1.5445
k1 = 2*np.pi*n1/1064e-9
k2 = 2*np.pi*n2/582e-9
delta_k = k2 - 2*k1


e0 = 6140032    #V/m
tau = 5e-9
t = np.linspace(-5e-8, 5e-8, 10)
omega = 2*(np.pi)*C/wl
rp_at_1050 = np.radians(6732.5)      #in rad/mm
l = 0.03
number_of_steps_z = 40
z = np.linspace(0, l, number_of_steps_z)
dz = z[1] - z[0]     #in mm
x = np.linspace(0, 0, number_of_steps_z)
y = np.linspace(0, 0, number_of_steps_z)


################################################################################################################
'define suceptibilty tensor'
################################################################################################################


d11 = 0.3e-12
d12 = -0.3e-12
d13 = 0
d14 = 0.008e-12
d15 = 0
d16 = 0
d21 = 0
d22 = 0
d23 = 0
d24 = 0
d25 = -0.008e-12
d26 = -0.3e-12
d31 = 0
d32 = 0
d33 = 0
d34 = 0
d35 = 0
d36 = 0

tensor = ([[d11, d12, d13, d14, d15, d16],
              [d21, d22, d23, d24, d25, d26],
              [d31, d32, d33, d34, d35, d36]])

tensor_d =  torch.tensor(tensor) 


################################################################################################################
'define electic_field tensor'
################################################################################################################

def e(e0t, z, rp_at_1050):
    e_x = float(e0*np.cos(rp_at_1050*z))
    e_y = float(e0*np.sin(rp_at_1050*z))
    e_z = float(0)
    e_xyz = torch.tensor([[e_x**2],
                 [e_y**2],
                 [e_z],
                 [2*e_y*e_z],
                 [2*e_x*e_z],
                 [2*e_x*e_y]])
    return e_xyz

################################################################################################################
'''Function Definitions for Laser Pulses'''
################################################################################################################


def e0t(t, tau, e0, omega):
    """Returns the electic field envelope of a Gaussian pulse with given parameters.

    Args:
        t : time
        tau : Intensity FWHM pulse width
        E0 : peak electric field strength
    """
    sigma_t =  tau / (2*np.sqrt(2*np.log(2)))
    return e0 * np.exp(-t**2 / (2*sigma_t)**2)*np.sin(omega*t)


################################################################################################################
'calculation'
################################################################################################################

P = torch.empty(number_of_steps_z,3,1, dtype = torch.complex64)

# Define the integral function
def integrand(z):
    return np.exp(1j * delta_k * z)

for i in range(number_of_steps_z):
    P[i] += (-1j*omega/n2*C)*integrand(i*dz)*(torch.matmul(tensor_d, e(e0, dz*i, rp_at_1050)))
    
    
Pxx = np.array(P[:,0,0])
Pyy = np.array(P[:,1,0])
Pzz = np.array(P[:,2,0])

#%%
################################################################################################################
'plotting'
################################################################################################################
plt.cla()
plt.plot(z, Pxx)

''' we ahve to solve this equation to obtain second harmonic efficiency
Eˆ(2ω, ℓ) = −(jωd(eff)/n2ωc0)*Eˆ2(ω)*ℓ ·[sin (∆k*ℓ/2)*∆k*ℓ/2]*e^(j∆kℓ/2)
'''




