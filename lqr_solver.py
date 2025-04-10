import control as ct
from control.statesp import StateSpace
import control.matlab as mt
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation


# M = 2.0
# m = 0.3
# L = 0.25
# r = 0.03
# g = 9.81

state = ['x','theta', 'x_dot', 'theta_dot']

m_body = 1.105 - 0.3
m_wheels = 0.3
g = 9.81
L = 0.3

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

Q = np.identity(4)*0.01
Q[0,0] = 1.0
Q[1,1] = 1.0
Q[2,2] *= 0.01
Q[3,3] *= 0.01
R = np.identity(1) * 2

K, S, E = ct.lqr(A, B, Q, R)

# Print K with all positives to match convention in MuJoCo and real life system
print("\nK")
print(np.array2string(np.abs(K), separator=', '))

print("\n\nS")
print(S)

print("\n\nE")
print(E)

C = np.identity(4)
D = np.array([[0], [0], [0], [0]])




sys = StateSpace(A-B*K, B, C, D)

plt.figure(figsize=(8, 8))

# y, t = mt.impulse(sys)
y, t = mt.step(sys)
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(t.T, y[:,i,0].T)
    plt.title(f"Response for {state[i]}")

plt.tight_layout()


# plt.figure(figsize=(8, 8))


# theta = 0
# x = 0

# x_start = x
# y_start = 0.0
# x_end = x + L*np.sin(theta)
# y_end = L*np.cos(theta)
# ctr = 0
# def update(frame):
#     global ctr
#     ctr += 1

#     x = y[ctr, 0, 0]
#     theta = y[ctr, 1, 0]*1.5

#     x_start = x
#     y_start = 0.0
#     x_end = x + L*np.sin(theta)
#     y_end = L*np.cos(theta)

#     plt.clf()
#     plt.plot([x_start, x_end], [y_start, y_end], 'k-', lw=2)
#     plt.plot(x_start, y_start, 'ko', ms=25)
#     plt.plot(x_end, y_end, 'bo', ms=50)


#     # Set the limits of the plot
#     plt.xlim(-2, 2)
#     plt.ylim(-0.01, 2*L)

#     # Set the title of the plot
#     plt.title("Segway @ timestep = " + str(ctr*dt))


# dt = 0.5
# duration = 10.0
# anim = FuncAnimation(plt.gcf(), update, frames=np.arange(0, duration, dt), interval=10)

# print(y[:,i,0])

plt.show()