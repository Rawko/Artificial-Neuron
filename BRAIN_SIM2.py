import numpy as np 
import matplotlib.pyplot as plt

# Time-varying input
time = np.arange(0,1000,1)
s = np.sin(1/15*time)

# Number of neurons
n_neurons = 2

# Initialize state variables for both neurons
vu = np.zeros((n_neurons, 1000))
uu = np.zeros((n_neurons, 1000))

# Parameters
a = 0.02
b = 0.2
c = -65
d = 8

# Initial conditions for both neurons
v = np.array([-65, -65])  # In millivolts
u = b * v  # Unitless

for t in range(1000):
    I = s[t] + 5 + np.random.normal(0,1, size=n_neurons)
    
    v = v + 0.5 * (0.04 * (v**2) + 5 * v + 140 - u + I)
    v = v + 0.5 * (0.04 * (v**2) + 5 * v + 140 - u + I)
    
    u = u + a * (b * v - u)
    
    vu[:, t] = v
    uu[:, t] = u
    
    spike_indices = np.where(v > 30)[0]
    v[spike_indices] = c
    u[spike_indices] = u[spike_indices] + d

# Plotting the membrane potentials of both neurons
plt.figure(figsize=(10, 6))
plt.plot(vu[0, :], label="Neuron 1")
plt.plot(vu[1, :], label="Neuron 2")
plt.legend()
plt.show()
