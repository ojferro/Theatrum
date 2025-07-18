import control as ct
from control.statesp import StateSpace
import control.matlab as mt
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

# Turn off scientific notation
np.set_printoptions(suppress=True)

# From: https://zumosegway.readthedocs.io/en/latest/model/

# Wheels
m_w = 0.150 * 2 # Mass of 2 wheels
R = 0.03 # Radius of wheel
I_w = m_w * R**2

# Body
m_b = 0.810 # Mass of body
L = 0.274 # Distance from wheel axle to CoM
I_b = m_b * L**2

# Damping
Beta_g = 0.0001 # Rolling damping ratio
Beta_m = 0.0001 # Bearing damping ratio

# Constants
g = 9.81


E = np.array([
    [(I_w + (m_w + m_b) * R**2), (m_b*R*L)],
    [(m_b*R*L), (I_b + m_b*L**2)]])

F = np.array([
    [(Beta_g + Beta_m), (-Beta_m)],
    [(-Beta_m), (Beta_m)]])

G = np.array([[0],[-m_b*g*L]])

H = [[1], [-1]]

# State [phi, theta, phi_dot, theta_dot]


E_inv = np.linalg.inv(E)


# State space matrices
A = np.zeros((4,4))
A[:2, 2:] = np.identity(2)
A[2:, 1:2] = -E_inv @ G
A[2:, 2:] = -E_inv @ F

# print("E_inv @ G")
# print(-E_inv @ G)
# print("E_inv @ F")
# print(-E_inv @ F)

B = np.zeros((4,1))
B[2:, :] = -E_inv @ H

C = np.identity(4)

D = np.zeros((4,1))

print("A")
print(A)

print("B")
print(B)

# Solve LQR
# print("Eigenvalues of Plant: ", np.linalg.eig(A)[0])

Q = np.identity(4)
# Q[0,0] = 0.0
# Q[1,1] = 1.0
# Q[2,2] = 1.0
# Q[3,3] = 1.0
R = np.identity(1)# * 2

K, S, E = ct.lqr(A, B, Q, R)

print("K")
print(K)

# Save np array K to file
np.save('K.npy', K)