import control as ct
from control.statesp import StateSpace
import control.matlab as mt
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation


M = 3.0
m = .5
L = 0.25
r = 0.03
g = 9.81

state = ['x','theta', 'x_dot', 'theta_dot']

# Constants
# gamma = (3*m+2*M)/3*m
# phi = 1/(3*L*m/2 + M*L) + 1/(M*L**2)

# c1 = M*gamma*g/(3*m/2 + M)
# c2 = (L*gamma*phi - 1/r) / (3*m/2 + M)
# c3 = gamma * g/L
# c4 = gamma * phi

# A = np.array([[0, 0, 1, 0],
#               [0, 0, 0, 1],
#               [0, c1, 0, 0],
#               [0, c3, 0, 0]])
# B = np.array([[0], [0], [c2], [c4]])

m_c = 1.5
m_p = 0.5
g = 9.81
L = 1
d1 = 0.01
d2 = 0.01

A = np.array([[0,0,1,0],
              [0,0,0,1],
              [0,g*m_p/m_c, -d1/m_c, -d2/(L*m_c)],
              [0, g*(m_c+m_p)/(L*m_c), -d1/(L*m_c), -d2*(m_c+m_p)/(L**2 *m_c * m_p)]
              ])
B = np.array([[0],[0], [1/m_c], [1/(L*m_c)]])

print("Eigenvalues of Plant: ", np.linalg.eig(A)[0])

Q = np.identity(4)
# Q[0,0] = 0.01
# Q[1,1] = 1.0
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






if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()