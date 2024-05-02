# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:28:27 2024

@author: akbar
"""

import numpy as np
import torch
import math
import scipy as sp
import scipy.constants as const
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, 
                             QHBoxLayout, QLabel, QCheckBox, QDialog, QLineEdit, QDialogButtonBox, 
                             QButtonGroup, QRadioButton)
import sys


class PhaseMatchingWindow(QDialog):
    
    # Define a signal to emit the phase matching settings
    phase_matching_settings_saved = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        
        super(PhaseMatchingWindow, self).__init__(parent)
        
        self.setWindowTitle("Phase Matching Settings")
        
        layout = QVBoxLayout()
        
        # Phase_matching_type
        self.phase_matching_type_buttons = self.add_radio_section(layout, "Phase Matching type:", ["Type-I", "Type-II"], "Type-I")        
        
        # Phase Matching Plane
        self.add_radio_section(layout, "Phase Matching Plane:", ["XY", "YZ", "ZX"], "XY")
        
        # Omega Polarisation
        self.omega_1_buttons = self.add_radio_section(layout, "Omega_1 Polarisation:", ["Ordinary", "Extra-ordinary"], "Ordinary")
        self.add_radio_section(layout, "Omega_2 Polarisation:", ["Ordinary", "Extra-ordinary"], "Ordinary")
        
        
        for button in self.phase_matching_type_buttons:
            button.clicked.connect(lambda state, button=button: self.disable_other_radios(button, self.omega_1_buttons))
        for button in self.phase_matching_type_buttons:
            button.clicked.connect(lambda state, button=button: self.disable_other_radios(button, self.omega_1_buttons))
        
        self.label = QLabel("Phase_matching_angle:")
        self.box_input_Phase_matching_angle = pg.SpinBox(value=30)
        layout.addWidget(self.label)
        layout.addWidget(self.box_input_Phase_matching_angle)
        
        warning = QLabel('for the changes to take affect, ok has to be pressed!!!!!')
        layout.addWidget(warning)
        
        # OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.ok_button_clicked)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        
        self.setLayout(layout)

        
    def ok_button_clicked(self):
        # Emit the signal with the selected values
        selected_values = self.get_selected_values()
        self.phase_matching_settings_saved.emit(selected_values)
        self.close()
        
    def disable_other_radios(self, clicked_button, radio_buttons):
        if self.get_selected_values()["phase_matching_type"] == "Type-II":
            for button in radio_buttons:
                if button is not clicked_button:
                    button.setEnabled(not clicked_button.isChecked())
        else:
            for button in radio_buttons:
                if button is not clicked_button:
                    button.setEnabled(clicked_button.isChecked())
        
    def add_radio_section(self, layout, label_text, options, default_option):
        radio_layout = QHBoxLayout()
        
        label = QLabel(label_text)
        radio_layout.addWidget(label)
        
        button_group = QButtonGroup(self)
        buttons = []
        
        for option in options:
            radio_button = QRadioButton(option)
            button_group.addButton(radio_button)
            radio_layout.addWidget(radio_button)
            buttons.append(radio_button)
            
            if option == default_option:
                radio_button.setChecked(True)
        
        layout.addLayout(radio_layout)
        return buttons
    
    def get_selected_values(self):
        
        selected_values = {
            "phase_matching_type": None,
            "phase_matching_plane": None,
            "pump_beam_polarisation": None,
            "omega_2_polarisation": None,
            "phase_matching_angle": None
        }
        phase_matching_plane_buttons = self.findChildren(QRadioButton)[0:2]
        for button in phase_matching_plane_buttons:
            if button.isChecked():
                selected_values["phase_matching_type"] = button.text()
        
        phase_matching_plane_buttons = self.findChildren(QRadioButton)[2:5]
        for button in phase_matching_plane_buttons:
            if button.isChecked():
                selected_values["phase_matching_plane"] = button.text()
        
        omega_1_buttons = self.findChildren(QRadioButton)[5:7]  # Assuming the order in the layout
        for button in omega_1_buttons:
            if button.isChecked():
                selected_values["pump_beam_polarisation"] = button.text()
        
        omega_2_buttons = self.findChildren(QRadioButton)[7:9]  # Assuming the order in the layout
        for button in omega_2_buttons:
            if button.isChecked():
                selected_values["omega_2_polarisation"] = button.text()
        
        selected_values["phase_matching_angle"] = float(self.box_input_Phase_matching_angle.value())
        print(selected_values)
        return selected_values


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Herriot Cell")
        self.phase_matching_settings = None  # Initialize phase matching settings
        self.GUI()
        
    def GUI(self):
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        
        
        self.left_layout = QVBoxLayout()

        
        #Create input labels and boxes using definition create_input('label', value) written below
        self.box_input_R1 = self.create_input("R1", 600)
        self.box_input_R2 = self.create_input("R2", 600)
        self.box_input_L = self.create_input("L", 1085)
        self.box_input_N = self.create_input("N", 5)

        self.box_input_crystal_thickness = self.create_input("Crystal Thickness", 1.5)
        self.box_input_crystal_position = self.create_input("Crystal Position", 542.5)
        self.box_input_pattern_size = self.create_input("Pattern Size", 15)

        self.box_input_λ1 = self.create_input("λ_pump_(in micro_meter)", 1.064)
        self.box_input_p = self.create_input("air_pressure", 101325)
        self.box_input_h = self.create_input("humidity(0 to 1)", 0)
        self.box_input_xc = self.create_input("co2 in ppm(225)",450)
        
        
        #button to open subwindow
        self.phase_matching_btn = QPushButton("Phase Matching Settings")
        self.left_layout.addWidget(self.phase_matching_btn)
        self.phase_matching_btn.clicked.connect(self.show_phase_matching_window)
        
        #create a checkbox for plotting
        self.checkbox = QCheckBox("view spot pattern")
        self.left_layout.addWidget(self.checkbox)
        self.checkbox.setChecked(True)
        
        
        # Create a calculate button and add connection
        self.calculate_btn = QPushButton("Calculate")
        self.quit_btn = QPushButton("Quit")
        self.left_layout.addWidget(self.calculate_btn)
        self.left_layout.addWidget(self.quit_btn)
        
        self.layout.addLayout(self.left_layout)
        
        
        # Create a horizontal layout to hold the PlotWidgets
        self.h_layout = QHBoxLayout()
        

        # Create the first PlotWidget
        self.plot_widget_1 = pg.PlotWidget()
        self.h_layout.addWidget(self.plot_widget_1)

        # Create the second PlotWidget
        self.plot_widget_2 = pg.PlotWidget()
        self.h_layout.addWidget(self.plot_widget_2)

        # Add the horizontal layout to the main layout
        self.layout.addLayout(self.h_layout)


        #actions on btns
        self.calculate_btn.clicked.connect(self.calculate_btn_clicked)
        self.quit_btn.clicked.connect(self.close_btn_clicked)

    def close_btn_clicked(self):
        QApplication.closeAllWindows()
        QApplication.quit()
        
    def save_phase_matching_settings(self, settings):
        # Store the phase matching settings
        self.phase_matching_settings = settings
        return self.phase_matching_settings
        
    def show_phase_matching_window(self):
        self.phase_matching_window = PhaseMatchingWindow(self)
        self.phase_matching_window.phase_matching_settings_saved.connect(self.save_phase_matching_settings)
        self.phase_matching_window.exec_()
        
    def create_input(self, label_text, default_value):
        # Create a label
        label = QLabel(f"{label_text}:")
        
        # Create a SpinBox
        spin_box = pg.SpinBox(value=default_value)
        
        # Add to layout
        self.left_layout.addWidget(label)
        self.left_layout.addWidget(spin_box)
        
        return spin_box
        
    def calculate_btn_clicked(self):
        beam_type = self.phase_matching_settings['pump_beam_polarisation']
        plane = self.phase_matching_settings['phase_matching_plane']
        phase_matching_angle = self.phase_matching_settings["phase_matching_angle"]
        R1 = float(self.box_input_R1.value())
        R2 = float(self.box_input_R2.value())
        L = float(self.box_input_L.value())
        N = int(self.box_input_N.value())
        crystal_length = float(self.box_input_crystal_thickness.value())
        crystal_position = float(self.box_input_crystal_position.value())
        pattern_size = float(self.box_input_pattern_size.value())
        λ1 = float(self.box_input_λ1.value())
        p = float(self.box_input_p.value())
        h = float(self.box_input_h.value())
        xc = float(self.box_input_xc.value())
        phase_matching_type = self.phase_matching_settings['phase_matching_type']
        first_mirror, second_mirror = simulate_travel(R1, R2, N, L, crystal_length, crystal_position, pattern_size,                             #cell geometry
                                                      λ1, p, h, xc,                                                                             #air properties
                                                      beam_type, plane, phase_matching_angle, phase_matching_type).spot_data()                  #phase_matching_settings
        
        
        if self.checkbox.isChecked():
            # Clear anz existing plots
            self.plot_widget_1.clear()
            self.plot_widget_2.clear()
            
            # Plot the first mirror spot pattern on the first PlotWidget
            self.plot_widget_1.plot(first_mirror[:,0], first_mirror[:,1], pen=None, symbol='o', symbolPen='b', symbolSize=10)
            
            # Plot the second mirror spot pattern on the second PlotWidget
            self.plot_widget_2.plot(second_mirror[:,0], second_mirror[:,1], pen=None, symbol='x', symbolPen='r', symbolSize=10)
            
            # Set plot labels and title for the first PlotWidget
            self.plot_widget_1.setLabel('left', 'y-axis in mm')
            self.plot_widget_1.setLabel('bottom', 'x-axis in mm')
            self.plot_widget_1.setTitle('First Mirror Spot pattern')
            
            # Set plot labels and title for the second PlotWidget
            self.plot_widget_2.setLabel('left', 'y-axis in mm')
            self.plot_widget_2.setLabel('bottom', 'x-axis in mm')
            self.plot_widget_2.setTitle('Second Mirror Spot pattern')
        else:
            self.plot_widget_1.clear()
            self.plot_widget_2.clear()

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

# =============================================================================
# reason to keep calculation of selmire equations separate is to be able to change crystal when ever needed without a problem
# =============================================================================
class LBO_refractive_index:
    
    def __init__(self):
        pass
# =============================================================================
# refractive inex calculation of LBO
# =============================================================================

    def nx(self, λ):                                            #--> n(α)
        nx = np.sqrt(2.45768 + 0.0098877 /
                     (λ**2 - 0.026095) - 
                     0.013847 * λ**2)
        return nx
    '''here nx and ny are changed to transform the crstal system to the cell system'''
    
    def ny(self, λ):                                            #--> n(β)
        ny = np.sqrt(2.52500 + 0.017123 / 
                     (λ**2 + 0.0060517) - 
                     0.0087838 * λ**2)
        return ny
    
    def nz(self, λ):                                            #--> n(γ)
        nz = np.sqrt(2.58488 + 0.012737 / 
                     (λ**2 - 0.021414) - 
                     0.016293 * λ**2)
        return nz
#%%
class chi_tensor_for_LBO:
    def __init__(self):
        
        self.d11 = 0.3e-12
        self.d12 = -0.3e-12
        self.d13 = 0
        self.d14 = 0.008e-12
        self.d15 = 0
        self.d16 = 0
        self.d21 = 0
        self.d22 = 0
        self.d23 = 0
        self.d24 = 0
        self.d25 = -0.008e-12
        self.d26 = -0.3e-12
        self.d31 = 0
        self.d32 = 0
        self.d33 = 0
        self.d34 = 0
        self.d35 = 0
        self.d36 = 0
        
        
    def chi2_tensor(self):
            
        self.tensor = [[self.d11, self.d12, self.d13, self.d14, self.d15, self.d16],
                      [self.d21, self.d22, self.d23, self.d24, self.d25, self.d26],
                      [self.d31, self.d32, self.d33, self.d34, self.d35, self.d36]]
        self.tensor = np.asarray(self.tensor)
        return self.tensor
    
    def initial_vector(self, e0, theta, phi):

         e_x = e0 *np.sin(theta) *np.cos(phi)
         e_y = e0 *np.sin(theta) *np.sin(phi)
         e_z = e0 *np.cos(theta)
         return np.array([[e_x], [e_y], [e_z]])

    def trick(self, plane):
        e_x, e_y, e_z = self.initial_vector(1, np.pi/2, np.pi/2)[0][0], self.initial_vector(1, np.pi/2, np.pi/2)[1][0], self.initial_vector(1, np.pi/2, np.pi/2)[2][0]
        if plane == 'XY':            
            # Calculations for N2x
            self.e_z_transformed = e_x
        
            # Calculations for N2y
            self.e_y_transformed = e_y
            
            # Calculations for N2z
            self.e_x_transformed = e_z
            
        elif plane == 'YZ':
            # Calculations for N2x
            self.e_x_transformed = e_x
        
            # Calculations for N2y
            self.e_y_transformed = e_y
            
            # Calculations for N2z
            self.e_z_transformed = e_z
            
        elif plane == 'ZX':
            # Calculations for N2x
            self.e_y_transformed = e_x
        
            # Calculations for N2y
            self.e_x_transformed = e_y
            
            # Calculations for N2z
            self.e_z_transformed = e_z
            
        return [[self.e_x_transformed], [self.e_y_transformed], [self.e_z_transformed]]

    def e(self, e_0, theta, phi):
        e_x, e_y, e_z = self.trick(plane, e_x, e_y, e_z)
        
        e_x = float(e0*np.cos(self.theta))
        e_y = float(e0*np.sin(self.theta))
        e_z = float(0)
        e_xyz = torch.tensor([[e_x**2],
                              [e_y**2],
                              [e_z**2],
                              [2*e_y*e_z],
                              [2*e_x*e_z],
                              [2*e_x*e_y]])
        return e_xyz
    
###now use this class in propogation class at each pass to calculate polarisation
##additionally schrodinger equation in comoving frame can be solved
#%%
    
    
class refractive_index:
    def __init__(self):
        self.N = LBO_refractive_index()
        pass
    
    def calculate_refractive_indices(self, λ, plane):              #λ in micrometers
    
        ''' In this calculations travel direction is always assumed to be z and x is verical and y is horizontal so, rotate the obeject(in this case ellipse),
            according to the phase matching plane and always treat Theta as phase matching angle no matter what
            whaichever plane we chose we keep it p polarised (to the ground).
            
            #########################
            for yz plane keep all same,
            for xy plane nx -->  n(γ), nz --> n(α)
            for xz plane nx -->  n(β), ny --> n(α)
            
            #########################
            for extraordinary phi is always pi/20
            for ordinary phi is always pi/20
            '''
        if plane == 'XY':            
            # Calculations for N2x
            self.Nz = self.N.nx(λ)
        
            # Calculations for N2y
            self.Ny = self.N.ny(λ)
            
            # Calculations for N2z
            self.Nx = self.N.nz(λ)
            
        elif plane == 'YZ':
            # Calculations for N2x
            self.Nx = self.N.nx(λ)
        
            # Calculations for N2y
            self.Ny = self.N.ny(λ)
            
            # Calculations for N2z
            self.Nz = self.N.nz(λ)
            
        elif plane == 'ZX':
            # Calculations for N2x
            self.Ny = self.N.nx(λ)
        
            # Calculations for N2y
            self.Nx = self.N.ny(λ)
            
            # Calculations for N2z
            self.Nz = self.N.nz(λ)
            
        return  self.Nx, self.Ny, self.Nz
        
    def calculate_neff(self, λ, plane, θ, φ):
        Nx, Ny, Nz = self.calculate_refractive_indices(λ, plane)
        cos_φ = np.cos(φ)
        sin_φ = np.sin(φ)
        cos_θ = np.cos(θ)
        sin_θ = np.sin(θ)
        
        numerator = Nx**2 * Ny**2 * Nz**2
        denominator_term1 = Nz**2 * (Ny**2 * cos_φ**2 + Nx**2 * sin_φ**2) * sin_θ**2
        denominator_term2 = Nx**2 * Ny**2 * cos_θ**2
        
        neff = np.sqrt(numerator / (denominator_term1 + denominator_term2))
        print(neff)
        
        return neff



class abcd_matrices:
    
    def initial_condition(self, x0, y0, xd, yd):
        M_in = [[x0],
                [y0],
                [xd],
                [yd]]
        return np.array(M_in)
    
    def mirror_transformed_vector(self):
        M_mrr = [[-1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, 1]] 
        return M_mrr
    
    def free_space_matrix(self, d):
        M_d = [[1, 0, d, 0],
              [0, 1, 0, d],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]
        return np.array(M_d)
    
    def matrix_for_reflection(self, R):
        M_fm = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [-2/R, 0, 1, 0],
                [0, -2/R, 0, 1]]
        return np.array(M_fm)

    
    def matrix_for_refraction(self, bool, n_theta_1, n_theta_2, n_phi_1, n_phi_2):
        
        if bool == True:
           M_rf = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, n_theta_1/n_theta_2, 0],
                   [0, 0, 0, n_phi_1/n_phi_2]]
        else:
            M_rf = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, n_theta_2/n_theta_1, 0],
                    [0, 0, 0, n_phi_2/n_phi_1]]
        
        return np.array(M_rf)
    
    def theta_and_phi_operator(self):
        op = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]
        return np.array(op)
    
#%%
class simulate_travel:
    def __init__(self, R1, R2, N, cell_length, crystal_length, crystal_position, pattern_size, λ1, p, h, xc, pump_beam_type, plane, phase_matching_angle, phase_matching_type):
        self.λ1 = λ1
        self.N = N  # number of bounces on one mirror
        self.R1 = R1  # radius of curvature
        self.R2 = R2
        self.cell_length = cell_length      #969.84
        self.crystal_length = crystal_length    #1.5
        self.crystal_position = crystal_position    #542.5                                                     
        self.d1 = self.crystal_position - (self.crystal_length / 2)
        self.d2 = self.cell_length - (self.d1 + (self.crystal_length / 2))
        self.f = self.R1 / 2  # focal length
        self.M = self.N - 1  # parameter M   M=N-1 means the longest configuration
        self.pattern_size = pattern_size
        self.beam_type = pump_beam_type 
        self.plane = plane
        self.phase_matching_angle = phase_matching_angle
        self.phase_matching_type = phase_matching_type
        
        #second_harmonic_information
        self.λ2 = λ1 / 2
        #self.SH_beam_type = SH_beam_type
        
        #initial_coordinates
        
        self.x0 = self.pattern_size * 0.5 * np.sqrt(self.cell_length / self.f)
        self.y0 = self.x0 * np.sqrt(4 * self.f / self.cell_length - 1) 
        
        #initial_slopes
        self.kx = -1
        self.ky = 0
        self.xslope = self.kx * self.pattern_size / np.sqrt(self.R1 * self.cell_length / 2)
        self.yslope = self.ky * self.pattern_size / np.sqrt(self.R1 * self.cell_length / 2)
        
        
        self.matrices = abcd_matrices()
        #self.M_in = self.matrices.initial_condition(x0, y0, xd, yd)
        self.M_d1 = self.matrices.free_space_matrix(self.d1)
        self.M_d2 = self.matrices.free_space_matrix(self.d2)
        self.M_in_crystal = self.matrices.free_space_matrix(self.crystal_length)
        self.M_mrr = self.matrices.mirror_transformed_vector()
        self.M_f1m = self.matrices.matrix_for_reflection(self.R1)
        self.M_f2m = self.matrices.matrix_for_reflection(self.R2)
        
        
        self.n_air_p = Air_refractive_index()
        self.n_air_at_p = self.n_air_p.n(λ1, p, h, xc)
        self.refractive_index = refractive_index()
        
        
        #plt.scatter(self.x0, self.y0)
        self.abcd_matrices = abcd_matrices()
        self.propogation()
        #self.graph_pattern()

# =============================================================================
# travel fundamental 
# =============================================================================

    '''simiilaraly travel for the second harmonic can be defined ater the first pass, making spot pattern foir SH would help visualise the proccess!'''

    def propogation(self):
        self.travel_data = np.zeros(shape=((self.N*4)+1, 4, 1))
        initial_matrix = self.abcd_matrices.initial_condition(self.x0, self.y0, self.xslope, self.yslope)
        self.travel_data[0] = initial_matrix
        self.second_harmonic_beam = []
        self.polarisation_data = []
        self.relative_phase_data = []
        

        for i in range(0, self.N):
            
    # =============================================================================
    #       d1 travel 1
    # =============================================================================
                
            d1travel1 = self.after_d1_travel_in_air(initial_matrix)
            
    # =============================================================================
    #       operator toextract angles and then calculate refractive index
    # =============================================================================
            
            refr1 = self.after_refraction(d1travel1, True)
            d1_crystal = self.after_dcrystal_travel(refr1)
            
    # =============================================================================
    #       operator toextract angles and then calculate refractive index  
    # =============================================================================
            
            refr2 = self.after_refraction(d1_crystal, False)
            refr2 = d1travel1
            
    # =============================================================================
    #       d2 travel 1
    # =============================================================================
    
            d2travel1 = self.after_d2_travel_in_air(refr2)
            self.travel_data[i*4+1] = d2travel1
            
    # =============================================================================
    #       reflection from m2
    # =============================================================================
            
            reflected = self.after_reflection_from_m2(d2travel1)
            self.travel_data[i*4+2] = reflected
            
    # =============================================================================
    #       travel2
    # =============================================================================
    
            d2travel2 = self.after_d2_travel_in_air(reflected)

    # =============================================================================
    #       operator toextract angles and then calculate refractive index          
    # =============================================================================
    
            refr3 = self.after_refraction(d2travel2, True)
            d2_crystal = self.after_dcrystal_travel(refr3)
            
    # =============================================================================
    #       operator toextract angles and then calculate refractive index      
    # =============================================================================
    
            refr4 = self.after_refraction(d2_crystal, False)
            refr4 = d2travel2
            d1travel2 = self.after_d1_travel_in_air(refr4)
            
            self.travel_data[i*4+3] = d1travel2
            
    # =============================================================================
    #       reflection from m1
    # =============================================================================
            
            mirrortransformed = self.mirror_transformation(d1travel2)
            reflected2 = self.after_reflection_from_m1(mirrortransformed)
            reflected2 = self.mirror_transformation(reflected2)
            
            self.travel_data[i*4+4] = reflected2
            
    # =============================================================================
    #       loop statement
    # =============================================================================
    
            initial_matrix = reflected2
        
        return self.travel_data
    
    
    # =============================================================================
    #   calculate_phase_difference
    # =============================================================================
    def calculate_phase_difference(self, n1, n2, d):
        dk = (2*np.pi/self.λ2)*(n1-n2)
        d_psi = dk*d
        return d_psi
    
    # =============================================================================
    #   saving and organising the travel data
    # =============================================================================
        
    def spot_data(self):
        first_mirror = np.zeros(shape=(self.N,2))
        second_mirror = np.zeros(shape=(self.N,2))
        
        for i in range (0, self.N):    
            first_mirror[i] = self.travel_data[i*4,0:2,0]
            second_mirror[i] = self.travel_data[(i*4)+2,0:2,0]
            
        return first_mirror, second_mirror
    
    # =============================================================================
    #   just for testing purposes 
    # =============================================================================
    
    def graph_pattern(self):
        first_mirror, second_mirror = self.spot_data()
        
        plt.figure()
        plt.scatter(first_mirror[:,0], first_mirror[:,1])
        
        plt.figure()
        plt.scatter(second_mirror[:,0], second_mirror[:,1])
        
    # =============================================================================
    #   define operations
    # =============================================================================
    def after_d1_travel_in_air(self, M_in):
        return self.M_d1 @ M_in
    
    def after_d2_travel_in_air(self, M_in):
        return self.M_d2 @ M_in
    
    def after_dcrystal_travel(self, M_in):
        return self.M_in_crystal @ M_in
    
    def after_reflection_from_m1(self, M_in):
        return self.M_f1m @ M_in
    
    def after_reflection_from_m2(self, M_in):
        return self.M_f2m @ M_in
    
    def mirror_transformation(self, M_in):
        return self.M_mrr @ M_in
    
    def after_refraction(self, M_in, bool):         #if bool = true {air to crystal} else {crystal to air}
        polarisation = self.polarisation_calculation(M_in)
        self.neff = self.calculate_refractiv_index(polarisation)[0]                           #[0] to deal with the dimentions only
        print(self.neff)
        return self.matrices.matrix_for_refraction(bool, self.n_air_at_p, self.neff, self.n_air_at_p, self.neff) @ M_in
    
    
# =============================================================================
#       refractive index calculation
# =============================================================================

    '''refractive index calculation depending on theta, phi, type of beam'''
    
    def polarisation_calculation(self, M_in):
        result = self.matrices.theta_and_phi_operator()@M_in                                         #######put your phase mathing angle here
        x, y, theta, phi = result[0][0], result[1][0], result[2][0], result[3][0]
        
        ''' Just a note: 
                        for ordinary beam rotation in non phase_matching plane is affective(resulting in RI change) but,
                        usually not much beacuse it is close to (N*pi/2) rad on the ellipse.
                        for extraordinary beam rotation in phase_matching plane is affective(resulting in RI change). 
                        significant in this case because it is near phase matching angle.
        '''
        
        if self.phase_matching_type == 'Type-I':
            
            if self.beam_type == 'Ordinary':
                
                theta_transformed = np.pi/2
                phi_transformed = phi
                
            elif self.beam_type == 'Extra-ordinary':
                
                theta_transformed = np.pi/2+self.phase_matching_angle+theta
                phi_transformed = np.pi/2
                
        elif self.phase_matching_type == 'Type-II':
            
            theta_transformed = np.pi/2+self.phase_matching_angle+theta
            phi_transformed = np.pi/4+phi
            
        else:
            print('error: put beam type')
            
        print(theta_transformed, phi_transformed)
        theta_transformed = float(theta_transformed)
        phi_transformed = float(phi_transformed)
        M_out = [[x], [y], [theta_transformed], [phi_transformed]]
        
        return np.array(M_out)
    
    def calculate_refractiv_index(self, M_in):
        result = self.matrices.theta_and_phi_operator()@M_in
        theta_transformed, phi_transformed = result[2], result[3]
        
        '''here we use a trick to transofrm the cell system to the crystal system just by replacing nx and ny 
        (rotating the elipsoid instead of coordinate system whichis esssentially the same)'''
        n = self.refractive_index.calculate_neff(self.λ1, self.plane, theta_transformed, phi_transformed)
        
        return n
    # =============================================================================
    # phase shift calculations
    # =============================================================================
    
    def calculate_phase_shift(self, lambda1, lambda2, d, theta, phi):
        result = self.matrices.theta_and_phi_operator()@M_in
        travel_length = d*np.sqrt((1/np.cos(theta)**2)+(1/np.cos(phi)**2))
        dk = calculate_refractiv_index(self, M_in, beam_type, plane)
        dpsi = dk*travel_length

#%%
# =============================================================================
# class calculation_for_second_harmonic:
#     
#     def __init__(self, M_in, d, beam_type):      
#         ################################################################################################################
#         'parametrs'
#         ################################################################################################################
#         
#         
#         '''for qrartz d11 = 0.3 (pm/V) d14 = 0.008 (pm/V)
#         in imperical units d11 = 0.3e-12 (m/V) d14 = 0.008e-12 (m/V)'''
#         
#         self.M_in = self.simulate_travel.polarisation_calculation(M_in)
#         self.theta = theta
#         self.phi = phi
#         self.C = const.c
#         self.k1 = 2*np.pi*n1/1064e-9
#         self.k2 = 2*np.pi*n2/582e-9
#         self.delta_k = k2 - 2*k1
#         
#         self.beam_type = beam_type
#         self.e0 = 6140032    #V/m
#         self.tau = 5e-9
#         self.t = np.linspace(-5e-8, 5e-8, 10)
#         self.omega = 2*(np.pi)*C/wl
#         self.number_of_steps_z = 40
#         self.z = np.linspace(0, d, number_of_steps_z)
#         self.dz = z[1] - z[0]     #in mm
#         self.x = np.linspace(0, 0, number_of_steps_z)
#         self.y = np.linspace(0, 0, number_of_steps_z)
#         
#         
#     def polarisation_calculation(self, M_in):
#         result = self.matrices.theta_and_phi_operator()@M_in                                         #######put your phase mathing angle here
#         x, y, theta, phi = result[0], result[1], result[2], result[3]
#         
#         ''' Just a note: 
#                         for ordinary beam rotation in non phase_matching plane is affective(resulting in RI change) but,
#                         usually not much beacuse it is close to (N*pi/2) rad on the ellipse.
#                         for extraordinary beam rotation in phase_matching plane is affective(resulting in RI change). 
#                         significant in this case because it is near phase matching angle.
#         '''
#             
#         if self.beam_type == 'Ordinary':
#             
#             theta_transformed = np.pi/2
#             phi_transformed = phi
#             
#         elif self.beam_type == 'Extra-ordinary':
#             
#             theta_transformed = np.pi/2+self.phase_matching_angle+theta
#             phi_transformed = np.pi/2
# 
#         else:
#             print('error: put beam type')
#             
#         print(theta_transformed, phi_transformed)
#         M_out = [[x],
#                 [y],
#                 [theta_transformed],
#                 [phi_transformed]]
#         M_in = np.array(M_in)
#         return np.array(M_in)
# 
#     ################################################################################################################
#     'define electic_field tensor'
#     ################################################################################################################
#     
# 
# 
#     ################################################################################################################
#     '''Function Definitions for Laser Pulses'''
#     ################################################################################################################
#     
#     
#     def e0t(self, t, tau, e0, omega):
#         """Returns the electic field envelope of a Gaussian pulse with given parameters.
#     
#         Args:
#             t : time
#             tau : Intensity FWHM pulse width
#             E0 : peak electric field strength
#         """
#         sigma_t =  tau / (2*np.sqrt(2*np.log(2)))
#         return e0 * np.exp(-t**2 / (2*sigma_t)**2)*np.sin(omega*t)
# 
# 
#     ################################################################################################################
#     'calculation'
#     ################################################################################################################
#     
#     P = torch.empty(number_of_steps_z,3,1, dtype = torch.complex64)
#     
#     # Define the integral function
#     def integrand(self, z):
#         return np.exp(1j * delta_k * z)
#     
#     for i in range(number_of_steps_z):
#         P[i] += (-1j*omega/n2*C)*integrand(i*dz)*(torch.matmul(tensor_d, e(e0, dz*i, rp_at_1050)))
# =============================================================================
        

#%%
        
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
window.close()


# =============================================================================
'problems & notes right now'
# =============================================================================
# there scemes to be no dependancy over refractive index of either air or nl crystal which we want to include  --- problem solved
# after operator operate on the inpuut 4*1 vector they should not change for the next opticxal element operation   --- problem solved
# write a function that goes after the surface refraction part and solve the electric filed equations and manley-rowe relations  --- going to start soon

# 26/04/2024 update:(Notes)
# having problems withh selcting the beam type and which beam from the selection menu function over there are unfinished so be careful  ---solved(29/04/2024)
# lets calculate fundamental beam first and we cn proceed
# idea is good but we need correct way to store beam data so function can use it even in loop without going default  ---solved(29/04/2024)
# may be one solution could be to load data in __init__ of silulate travel and then can be used efficiently.
# make refracive index calculation and every function in that class wavelength dependent and the from __init__ we can use it... (Enjoy the weekend!)



### write difffernet function inside the main propogation loop that will append values in each diffrerent catagoris such as relative phase and coordinates
# 26/04/2024 update:(Notes):
    #I have theta and phi in spherical coordinate i just need to rotate the crystal in the correct direction depending on the plane