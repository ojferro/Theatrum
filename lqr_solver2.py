import control as ct
from control.statesp import StateSpace
import control.matlab as mt
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

# Turn off scientific notation
np.set_printoptions(suppress=True)

# M = 2.0
# m = 0.3
# L = 0.25
# r = 0.03
# g = 9.81

state = ['x','theta', 'x_dot', 'theta_dot']

est_battery_mass = 0.4
m_body = 1.105 - 0.3 #+ est_battery_mass
m_wheels = 0.3
g = 9.81
L = 0.2742031409437471

# Damping coeffs
d1 = 0.001
d2 = 0.001

A = np.array([[0,0,1,0],
              [0,0,0,1],
              [0,g*m_wheels/m_body, -d1/m_body, -d2/(L*m_body)],
              [0, g*(m_body+m_wheels)/(L*m_body), -d1/(L*m_body), -d2*(m_body+m_wheels)/(L**2 *m_body * m_wheels)]
              ])
B = np.array([[0],[0], [1/m_body], [1/(L*m_body)]])

print("Eigenvalues of Plant: ", np.linalg.eig(A)[0])

Q = np.identity(4)
Q[0,0] = 0.0
Q[1,1] = 1.0
Q[2,2] = 4.0
Q[3,3] = 1.0
R = np.identity(1) * 2

K, S, E = ct.lqr(A, B, Q, R)

# Print K with all positives to match convention in MuJoCo and real life system
print("\nK")
print(np.array2string(np.abs(K), separator=', '))

# print("\n\nS")
# print(S)

# print("\n\nE")
# print(E)

C = np.identity(4)
D = np.array([[0], [0], [0], [0]])




sys = StateSpace(A-B*K, B, C, D)

plt.figure(figsize=(8, 8))

y, t = mt.step(sys)
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(t.T, y[:,i,0].T)
    plt.title(f"Response for {state[i]}")

plt.tight_layout()
# plt.show()