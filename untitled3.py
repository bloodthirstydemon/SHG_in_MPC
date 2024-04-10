# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:28:27 2024

@author: akbar
"""

import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk


class MainWindow:
    def __init__(self):
        self.window = Tk()
        self.window.title("Herriot Cell")
        self.window.configure(background="grey")
        self.window.geometry("1300x900")
        self.canvas = tk.Canvas(self.window)
        self.canvas.pack(side=tk.RIGHT, padx=10) 

        # Label and input box for R1
        self.label_R1 = tk.Label(self.window, text="R1:")
        self.label_R1.pack(side=tk.LEFT)
        self.box_input_R1 = tk.Text(self.window, state=tk.NORMAL, height=1, width=5)
        self.box_input_R1.pack(side=tk.LEFT)

        # Label and input box for R2
        self.label_R2 = tk.Label(self.window, text="R2:")
        self.label_R2.pack(side=tk.LEFT)
        self.box_input_R2 = tk.Text(self.window, state=tk.NORMAL, height=1, width=5)
        self.box_input_R2.pack(side=tk.LEFT)

        # Label and input box for cell length
        self.label_L = tk.Label(self.window, text="L:")
        self.label_L.pack(side=tk.LEFT)
        self.box_input_L = tk.Text(self.window, state=tk.NORMAL, height=1, width=5)
        self.box_input_L.pack(side=tk.LEFT)
        
        # Label and input box for number of bounces
        self.label_N = tk.Label(self.window, text="N:")
        self.label_N.pack(side=tk.LEFT)
        self.box_input_N = tk.Text(self.window, state=tk.NORMAL, height=1, width=5)
        self.box_input_N.pack(side=tk.LEFT)
        
        # Label and input box for crystal position
        self.label_crystal_position = tk.Label(self.window, text="crystal position:")
        self.label_crystal_position.pack(side=tk.LEFT)
        self.box_input_crystal_position = tk.Text(self.window, state=tk.NORMAL, height=1, width=5)
        self.box_input_crystal_position.pack(side=tk.LEFT)
        
        # Label and input box for crystal thickness
        self.label_crystal_thickness = tk.Label(self.window, text="crystal thickness:")
        self.label_crystal_thickness.pack(side=tk.LEFT)
        self.box_input_crystal_thickness = tk.Text(self.window, state=tk.NORMAL, height=1, width=5)
        self.box_input_crystal_thickness.pack(side=tk.LEFT)

        # Button to run programm
        self.calculate_btn = tk.Button(text="Calculate", width=12, command=self.Calculate_btn)
        self.calculate_btn.pack()

        # Quit button for main window
        self.btn_quit = tk.Button(self.window, text="Quit", command=self.quit)
        self.btn_quit.pack()
        #self.plot()
        # Main loop
        self.window.mainloop()
        
    def validate_input(new_text):
        if new_text.isdigit() or new_text == "":
            return True
        else:
            tk.messagebox.showerror("Error", "Please enter only numeric values.")
            return False
        
    def Calculate_btn(self):
        R1 = int(self.box_input_R1.get("1.0", "end-1c"))
        R2 = int(self.box_input_R2.get("1.0", "end-1c"))
        L = int(self.box_input_L.get("1.0", "end-1c"))
        xxx = cell_stability(R1, R2, L)
        self.canvas.delete("all")
        self.canvas.create_text(10, 10, anchor="nw", text=xxx.check_stability())
    
    
# =============================================================================
#     def plot(self):
#         b = HerriottCell(int(self.box_input_R1.get("1.0", "end-1c")), int(self.box_input_N.get("1.0", "end-1c")))
#         b.plot_pattern()
# =============================================================================
    
    def quit(self):
         self.window.destroy()

        
#%%
class cell_stability:
    def __init__(self, R1, R2, L):
        self.R1 = R1
        self.R2 = R2
        self.L = L
        self.g1 = 1 - (R1/L)
        self.g2 = 1 - (R2/L)
        
    def check_stability(self):
        if 0<=self.g1*self.g2<=1:
            return (f"cell/cavity is stable for R1={self.R1}mm, R2={self.R2}mm, L={self.L}mm")
        else:
            return ("Error: Unstable cell/cavity")
            
#%%
        
class HerriottCell:
    def __init__(self, ROC, N):
        self.ROC = ROC  # radius of curvature
        self.f = self.ROC / 2  # focal length
        self.N = N  # number of bounces on one mirror
        self.M = self.N - 1  # parameter M   M=N-1 means the longest configuration
        self.theta_HC = self.M * np.pi / self.N  # angle between bounces
        self.x = np.empty(2 * self.N + 1, dtype=object)  # x array creation
        self.y = np.empty(2 * self.N + 1, dtype=object)  # y array creation

    def beam_path(self):
        for count in range(2 * self.N + 1):  # x and y initial values
            self.x[count] = 0
            self.y[count] = 0

        self.A = 10  # Pattern's radius
        self.d = self.ROC * (1 - np.cos(self.theta_HC))  # distance between mirrors for the certain M

        self.x[0] = self.A * 0.5 * np.sqrt(self.d / self.f)  # input beam x coordinate
        self.y[0] = self.x[0] * np.sqrt(4 * self.f / self.d - 1)  # input beam y coordinate

        self.kx = -1  # sign of the x slope
        self.ky = 0  # sign of the y slope

        self.xslope = self.kx * self.A / np.sqrt(self.ROC * self.d / 2)  # x slope calculation
        self.yslope = self.ky * self.A / np.sqrt(self.ROC * self.d / 2)  # y slope calculation

        for count in range(2 * self.N):  # whole the X and Y coordinates array calculations
            self.x[count + 1] = self.x[0] * np.cos((count + 1) * self.theta_HC) + np.sqrt(
                self.d / (4 * self.f - self.d)) * (self.x[0] + 2 * self.f * self.xslope) * np.sin(
                (count + 1) * self.theta_HC)
            self.y[count + 1] = self.y[0] * np.cos((count + 1) * self.theta_HC) + np.sqrt(
                self.d / (4 * self.f - self.d)) * (self.y[0] + 2 * self.f * self.yslope) * np.sin(
                (count + 1) * self.theta_HC)

    def plot_pattern(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        for count in range(2 * self.N + 1):
            if count % 2 == 0:
                ax1.plot(self.x[count], self.y[count], 'or')
                ax1.annotate(str(count), (self.x[count], self.y[count]), fontsize=10)
            else:
                ax2.plot(self.x[count], self.y[count], 'ob')
                ax2.annotate(str(count), (self.x[count], self.y[count]), fontsize=10)

        ax1.set_xlim([-15, 15])
        ax1.set_ylim([-15, 15])
        ax1.set_title('Even Counts')
        ax1.set_aspect('equal')

        ax2.set_xlim([-15, 15])
        ax2.set_ylim([-15, 15])
        ax2.set_title('Odd Counts')
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.show()
        
#%%
class Air_refractive_index:
    
    def __init__(self):
        ##changeable variables
        self.T = 300
        
        ##fixed variables
        self.t = self.T-273.15
        self.a0 = 1.58123e-6   #K·Pa^-1
        self.a1 = -2.9331e-8   #Pa^-1
        self.a2 = 1.1043e-10   #K^-1·Pa^-1
        self.b0 = 5.707e-6     #K·Pa^-1
        self.b1 = -2.051e-8    #Pa^-1
        self.c0 = 1.9898e-4    #K·Pa^-1
        self.c1 = -2.376e-6    #Pa^-1
        self.d  = 1.83e-11     #K^2·Pa^-2
        self.e  = -0.765e-8    #K^2·Pa^-2
        
    def Z(self,p,xw): #compressibility

        return 1-(p/self.T)*(self.a0+self.a1*self.t+self.a2*self.t**2+(self.b0+self.b1*self.t)*xw+(self.c0+self.c1*self.t)*xw**2) + (p/self.T)**2*(self.d+self.e*xw**2)
    
    def n(self, λ,p,h,xc):
        # λ: wavelength, 0.3 to 1.69 μm 
        # t: temperature, -40 to +100 °C
        # p: pressure, 80000 to 120000 Pa
        # h: fractional humidity, 0 to 1
        # xc: CO2 concentration, 0 to 2000 ppm
    
        σ = 1/λ           #μm^-1
        R = 8.314510      #gas constant, J/(mol·K)
        
        k0 = 238.0185     #μm^-2
        k1 = 5792105      #μm^-2
        k2 = 57.362       #μm^-2
        k3 = 167917       #μm^-2
     
        w0 = 295.235      #μm^-2
        w1 = 2.6422       #μm^-2
        w2 = -0.032380    #μm^-4
        w3 = 0.004028     #μm^-6
        
        A = 1.2378847e-5  #K^-2
        B = -1.9121316e-2 #K^-1
        C = 33.93711047
        D = -6.3431645e3  #K
        
        α = 1.00062
        β = 3.14e-8       #Pa^-1,
        γ = 5.6e-7        #°C^-2
    
        #saturation vapor pressure of water vapor in air at temperature T
        if(self.t>=0):
            svp = np.exp(A*self.T**2 + B*self.T + C + D/self.T) #Pa
        else:
            svp = 10**(-2663.5/self.T+12.537)
        
        #enhancement factor of water vapor in air
        f = α + β*p + γ*self.t**2
        
        #molar fraction of water vapor in moist air
        xw = f*h*svp/p
        
        #refractive index of standard air at 15 °C, 101325 Pa, 0% humidity, 450 ppm CO2
        nas = 1 + (k1/(k0-σ**2)+k3/(k2-σ**2))*1e-8
        
        #refractive index of standard air at 15 °C, 101325 Pa, 0% humidity, xc ppm CO2
        naxs = 1 + (nas-1) * (1+0.534e-6*(xc-450))
        
        #refractive index of water vapor at standard conditions (20 °C, 1333 Pa)
        nws = 1 + 1.022*(w0+w1*σ**2+w2*σ**4+w3*σ**6)*1e-8
        
        Ma = 1e-3*(28.9635 + 12.011e-6*(xc-400)) #molar mass of dry air, kg/mol
        Mw = 0.018015                            #molar mass of water vapor, kg/mol
        
        Za = self.Z(101325, 0)                #compressibility of dry air
        Zw = self.Z(1333, 1)                  #compressibility of pure water vapor
        
        #Eq.4 with (T,P,xw) = (288.15, 101325, 0)
        ρaxs = 101325*Ma/(Za*R*288.15)           #density of standard air
        
        #Eq 4 with (T,P,xw) = (293.15, 1333, 1)
        ρws  = 1333*Mw/(Zw*R*293.15)             #density of standard water vapor
        
        # two parts of Eq.4: ρ=ρa+ρw
        ρa   = p*Ma/(self.Z(p,xw)*R*self.T)*(1-xw)       #density of the dry component of the moist air    
        ρw   = p*Mw/(self.Z(p,xw)*R*self.T)*xw           #density of the water vapor component
        
        nprop = 1 + (ρa/ρaxs)*(naxs-1) + (ρw/ρws)*(nws-1)
        
        return nprop


#%%
class LBO_refractive_index():
    
    def __init__(self):
        self.w1 = 1.030                              #wavelength in micro meter
        self.w2 = 0.515                              #wavelength in micro meter
        self.now = self.nx(self.w1)
        self.new = self.ny(self.w1)
        self.no2w = self.nx(self.w2)
        self.ne2w = self.ny(self.w2)
        self.crystal_length = 1500                   #crystal width in micro meter
        self.dtheta = 0.013824                       #incident angle of k and crystal plane
        self.theta = np.linspace(0, 2*np.pi, 1000)
        
        
# =============================================================================
# refractive inex calculation of LBO
# =============================================================================
        
    def nx(self, λ):
        nx = np.sqrt(2.45768 + 0.0098877 /
                     (λ**2 - 0.026095) - 
                     0.013847 * λ**2)
        return nx
    
    def ny(self, λ):
        ny = np.sqrt(2.52500 + 0.017123 / 
                     (λ**2 + 0.0060517) - 
                     0.0087838 * λ**2)
        return ny
    
    def nz(self, λ):
        nz = np.sqrt(2.58488 + 0.012737 / 
                     (λ**2 - 0.021414) - 
                     0.016293 * λ**2)
        return nz
        
# =============================================================================
# # =============================================================================
# # paper implementation
# # =============================================================================
# 
# 
#     def omega(self, wavelength):
#         omega = np.arcsin((self.nz(wavelength)/self.ny(wavelength))*
#                           (np.sqrt(((self.ny(wavelength)**2)-(self.nx(wavelength)**2))/((self.nz(wavelength)**2)-(self.nx(wavelength)**2)))))
#         return omega
#     
#     def thetadd(self, theta, phi):
#         thetadd = np.arctan(np.tan(theta)*np.cos(phi))
#         return thetadd
#         
#     def theta1(self, theta, phi, wavelength):    
#         theta1 = np.arccos((np.cos(theta)/
#                             np.cos(self.thetadd(theta, phi)))*np.cos(self.omega(wavelength)-self.thetadd(theta, phi)))
#         return theta1
#     
#     def theta2(self, theta, phi, wavelength):
#         theta2 = np.arccos((np.cos(theta)/
#                             np.cos(self.thetadd(theta, phi)))*np.cos(self.omega(wavelength)+self.thetadd(theta, phi)))
#         return theta2
# 
#     def neff(self, theta, phi, wavelength):
#         neff = np.sqrt(((self.nz(wavelength)**2)*(self.nx(wavelength)**2))/
#                        ((self.nx(wavelength)**2)*np.sin((self.theta1(theta, phi, wavelength)-self.theta2(theta, phi, wavelength))/2)**2+
#                         (self.nz(wavelength)**2)*np.cos((self.theta1(theta, phi, wavelength)-self.theta2(theta, phi, wavelength))/2)**2))
#         
#         return neff
# =============================================================================
       
    def polar_equation(self):
        return np.sqrt((self.new*self.now)**2/((np.cos(self.theta)**2 * self.new**2) 
                                               + (np.sin(self.theta)**2 * self.now**2)))
    
    def plot(self):
        # Generate theta values from 0 to 2*pi
        self.theta = np.linspace(0, 2*np.pi, 1000)

        # Calculate r values using the polar equation
        self.r = self.polar_equation()

        # Convert polar coordinates to Cartesian coordinates
        self.x = self.r * np.cos(self.theta)
        self.y = self.r * np.sin(self.theta)

        # Plot the ellipse
        plt.figure()
        plt.plot(self.x, self.y, label='Refractive_index for_y-z plane')
        plt.title('Refractive_index_Ellipse')
        plt.xlabel('n(x)')
        plt.ylabel('n(y)')
        plt.grid(True)
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.legend()
        plt.show()
# =============================================================================
# phase calculation
# =============================================================================

    def dk_due_to_incident_angle(self, phase_matching_angle):
        dk = ((2*np.pi*self.dtheta*np.sqrt(self.now)**3)/self.w1)*((1/self.no2w**2)
            -(1/self.ne2w**2))*np.sin(2*phase_matching_angle)
        return dk

    def accumulated_phase(self):
        DPhi = self.dk_due_to_incident_angle(phase_matching_angle)
        return DPhi


#%%

class phase_added_by_crystal:
    def __init__(self):
        pass
    def calculate_refractive_indices_of_LBO(self, λ):              #λ in micrometers
    
        # Calculations for N2x
        self.N2x = (2.45768 + 0.0098877 / (λ**2 - 0.026095) - 0.013847 * λ**2)
    
        # Calculations for N2y
        self.N2y = (2.52500 + 0.017123 / (λ**2 + 0.0060517) - 0.0087838 * λ**2)
        
        # Calculations for N2z
        self.N2z = (2.58488 + 0.012737 / (λ**2 - 0.021414) - 0.016293 * λ**2)
        
        return self.N2x, self.N2y, self.N2z
        
    def transformation_matrix_for_refractive_index(self, θ, φ):
        N2x, N2y, N2z = self.calculate_refractive_indices_of_LBO
    
        term1 = (N2y)*(N2z)*(np.sin(θ)**2 * np.cos(φ)**2)
        
        term2 = (N2x)*(N2z)*(np.sin(θ)**2 * np.sin(φ)**2)
        
        term3 = (N2x)*(N2y)*(np.cos(θ)**2)
        
        #neff calculation
        neff = np.sqrt((N2x)*(N2y)*(N2z)/ (term1 + term2 + term3))
        
        return neff
    
    def initial_condition(self, x0, y0, xd, yd):
        M_in = [[x0],
              [y0],
              [xd],
              [yd]]
        return np.array(M_in)
    
    def matrix_for_refraction(self, bool, n_theta_1, n_theta_2, n_phi_1, n_phi_2, n_air, n_crystal):
        
        if bool == 1:
            n_theta_1 = n_air
            n_theta_2 = n_crystal
        else:
            n_theta_1 = n_crystal
            n_theta_2 = n_air
            
        M_fm = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, n_theta_1/n_theta_2, 0],
                [0, 0, 0, n_phi_1/n_phi_2]]
        
        return np.array(M_fm)
    
    def matrix_for_reflection(self, R):
        M_fm = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [-2/R, 0, 1, 0],
                [0, -2/R, 0, 1]]
        return np.array(M_fm)
    
    def free_space_matrix(self, d):
        M_d = [[1, 0, d, 0],
              [0, 1, 0, d],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]
        return np.array(M_d)
    
class beam_travel:
    def __init__(self):
        self.N = 5  # number of bounces on one mirror
        self.ROC = 600  # radius of curvature
        self.cell_length = 1085
        self.crystal_length = 1.5
        self.crystal_position = 542.5                                                     
        self.d1 = self.crystal_position - (self.crystal_length / 2)
        self.d2 = self.cell_length - self.d1 - (self.crystal_length / 2)
        self.f = self.ROC / 2  # focal length
        self.M = self.N - 1  # parameter M   M=N-1 means the longest configuration
        self.A = 15
        
        #initial_coordinates
        
        self.x0 = self.A * 0.5 * np.sqrt(self.cell_length / self.f)
        self.y0 = self.x0 * np.sqrt(4 * self.f / self.cell_length - 1) 
        
        #initial_slopes
        self.kx = -1
        self.ky = 0
        self.xslope = self.kx * self.A / np.sqrt(self.ROC * self.cell_length / 2)
        self.yslope = self.ky * self.A / np.sqrt(self.ROC * self.cell_length / 2)
        #plt.scatter(self.x0, self.y0)
        self.propogation()

    def propogation(self):
        travel_data = []#np.zeros((4, 1, 16), dtype=float)
        a = phase_added_by_crystal()
        #initial_matrix = np.array(a.initial_condition(self.x0, self.y0, self.xslope, self.yslope))
        travel_data.append(initial_matrix)
        
# =============================================================================
#         lets calculate refraction in a type 1 phase matching crystal condition in LBO
# =============================================================================
        refractive_index = phase_added_by_crystal()
        
        refraction = a.matrix_for_refraction(n_theta_1, n_theta_2, n_phi_1, n_phi_2)
        
        
        for i in range(0, self.N):
            dtravel = np.array(a.free_space_matrix(self.d1))@initial_matrix
            travel_data.append(dtravel)
            initial_matrix = dtravel
        print(travel_data)





