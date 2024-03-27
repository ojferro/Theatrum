# Based on this paper: https://arxiv.org/ftp/arxiv/papers/2109/2109.11919.pdf
import numpy as np
import control as ct
from control.statesp import StateSpace
import control.matlab as mt
import matplotlib.pyplot as plt

state = ['x', 'x_dot', 'theta', 'theta_dot']


M = 1.0 # wheel mass
m = 3.0 # rod mass
l = 0.25 # rod length
R = 0.03 # wheel radius
Ir = 1/3.0 * m*l**2 #0.02667 # rod inertia
print(f"Ir = {Ir}")
Iw = 1/2.0 * M*R**2 #0.004375 # wheel inertia
print(f"Iw = {Iw}")
K = 1 # gear ratio for the wheel motors (effectively how much torque does the rod feel given wheel torque)

g = 9.81

k1 = (M+m)*R + (Iw/R)
k2 = m*l*R
k3 = m*l
k4 = Ir + m*l**2
k5 = m*g*l
k6 = M*l*R

A = np.array([[0,1,0,0],
          [0,0,-k2*k5/(k1*k4 - k2*k3), 0],
          [0,0,0,1],
          [0,0,-k1*k5/(k1*k4 - k2*k3), 0]])

B = np.array([[0],
              [(K*k2 + k4)/(k1*k4-k2*k3)],
              [0],
              [(K*k1 + k3)/(k1*k4-k2*k3)]])


print("Eigenvalues of Plant: ", np.linalg.eig(A)[0])

Q = np.identity(4)
# Q[0,0] = 0.01
# Q[1,1] *= 2.0
# Q[2,2] = 0.01
# Q[3,3] = 0.01
R = np.identity(1)

K, S, E = ct.lqr(A, B, Q, R)
print("\nK")
print(np.array2string(K, separator=', '))

print("\n\nS")
print(S)

print("\n\nE")
print(E)

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
plt.show()