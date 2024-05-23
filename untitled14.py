# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:27:09 2024

@author: akbar
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:54:32 2024

@author: akbar
"""

import numpy as np
import matplotlib.pyplot as plt

def dE2dz(z, E, delta_k, n2, λ, D):
    return -1j * ((2 * np.pi / λ * n2) * (D * fE2(E, delta_k, z)))

def dEdz(z, E2, E, delta_k, n, λ, D):
    return -1j * ((2 * np.pi / λ * n) * (D * fE(E2, E, delta_k, z)))

def fE2(E, delta_k, z):
    return E**2 * np.exp(1j * delta_k * z)

def fE(E2, E, delta_k, z):
    E21 = E2 * np.conj(E)
    return E21 * np.exp(-1j * delta_k * z)

def runge_kutta_step(z, E2, E, delta_k, n2, n, λ, D, dz):
    k1_E2 = dz * dE2dz(z, E, delta_k, n2, λ, D)
    k1_E = dz * dEdz(z, E2, E, delta_k, n, λ, D)
    
    k2_E2 = dz * dE2dz(z + dz/2, E + k1_E/2, delta_k, n2, λ, D)
    k2_E = dz * dEdz(z + dz/2, E2 + k1_E2/2, E + k1_E/2, delta_k, n, λ, D)
    
    k3_E2 = dz * dE2dz(z + dz/2, E + k2_E/2, delta_k, n2, λ, D)
    k3_E = dz * dEdz(z + dz/2, E2 + k2_E2/2, E + k2_E/2, delta_k, n, λ, D)
    
    k4_E2 = dz * dE2dz(z + dz, E + k3_E, delta_k, n2, λ, D)
    k4_E = dz * dEdz(z + dz, E2 + k3_E2, E + k3_E, delta_k, n, λ, D)
    
    E2_next = E2 + (k1_E2 + 2*k2_E2 + 2*k3_E2 + k4_E2) / 6
    E_next = E + (k1_E + 2*k2_E + 2*k3_E + k4_E) / 6
    
    relative_phase = delta_k * z
    
    return E2_next, E_next, relative_phase

def gaussian_pulse(t, t0, width, amplitude, omega):
    envelope = amplitude * np.exp(-(t - t0)**2 / (2 * width**2))
    pulse = envelope * np.exp(1j*omega*t)
    return pulse

def integrate_rk4(z_max, dz, t_max, dt, E2_0, E_0, delta_k, n2, n, λ, D):
    num_z_steps = int(z_max / dz)
    num_t_steps = int(t_max / dt)
    
    z_values = np.linspace(0, z_max, num_z_steps)
    t_values = np.linspace(-t_max/2, t_max/2, num_t_steps)
    
    E2_values = np.zeros((num_z_steps, num_t_steps), dtype=complex)
    E_values = np.zeros((num_z_steps, num_t_steps), dtype=complex)
    relative_phase_data = np.zeros((num_z_steps, num_t_steps), dtype=float)
    E_values[0, :] = E_0
    
    for i in range(1, num_z_steps):
        for j in range(num_t_steps):
            z = z_values[i-1]
            E2_next, E_next, relative_phase = runge_kutta_step(z, E2_values[i-1, j], E_values[i-1, j], delta_k, n2, n, λ, D, dz)
            E2_values[i, j] = E2_next
            E_values[i, j] = E_next
            relative_phase_data[i, j] = relative_phase
        total_energy = np.sum(np.abs(E_values[i, :])**2 + np.abs(E2_values[i, :])**2)
        print(f'Step {i}, Total Energy: {total_energy}')
    
    return z_values, t_values, E2_values, E_values, relative_phase_data

# Example usage
z_max = 1000.0  # Maximum value of z in micrometers
dz = 0.1  # Step size in micrometers
t_max = 10   # Temporal window in picoseconds
dt = 0.1     # Temporal step size in picoseconds
λ = 1.064    # Wavelength in micrometers
omega = 2*np.pi*299792458e-6/1.064
E_0_amplitude = 10000.0+0j
E_0_width = 2.0  # Pulse width in picoseconds
E_0_t0 = 0.0     # Pulse center in picoseconds
E_0 = gaussian_pulse(np.linspace(-t_max/2, t_max/2, int(t_max/dt)), E_0_t0, E_0_width, E_0_amplitude, omega)
E2_0 = np.zeros_like(E_0)

n = 1.5656
n2 = 1.5670#1.5785 + 0j
delta_k = (2 * np.pi / 0.515) * (n2 - n)

D = 0.85e-06#0.85e-06

z_values, t_values, E2_values, E_values, relative_phase_data = integrate_rk4(z_max, dz, t_max, dt, E2_0, E_0, delta_k, n2, n, λ, D)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(np.abs(E_values)**2, aspect='auto', extent=[t_values.min(), t_values.max(), z_values.min(), z_values.max()], origin='lower')
plt.colorbar(label='Intensity')
plt.title('Fundamental Pulse Evolution')
plt.xlabel('Time (ps)')
plt.ylabel('Propagation distance (µm)')

plt.subplot(2, 3, 2)
plt.imshow(np.abs(E2_values)**2, aspect='auto', extent=[t_values.min(), t_values.max(), z_values.min(), z_values.max()], origin='lower')
plt.colorbar(label='Intensity')
plt.title('Second Harmonic Pulse Evolution')
plt.xlabel('Time (ps)')
plt.ylabel('Propagation distance (µm)')

plt.subplot(2, 3, 3)
plt.imshow(relative_phase_data, aspect='auto', extent=[t_values.min(), t_values.max(), z_values.min(), z_values.max()], origin='lower')
plt.colorbar(label='relative phase in rad')
plt.title('phase of Pulse Evolution')
plt.xlabel('Time (ps)')
plt.ylabel('Propagation distance (µm)')

plt.subplot(2, 3, 4)
plt.plot(np.linspace(0, t_max, 100), np.abs(E_0)**2)
plt.title('Pulse envelope shape')
plt.xlabel('Time (ps)')
plt.ylabel('Propagation distance (µm)')

plt.subplot(2, 3, 5)
plt.plot(np.linspace(0, t_max, 100), E_0.real)
plt.title('electric field')
plt.xlabel('Time (ps)')
plt.ylabel('Propagation distance (µm)')

plt.tight_layout()
plt.show()
