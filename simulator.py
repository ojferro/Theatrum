import time

import mujoco
import mujoco.viewer

import numpy as np
import math
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def quaternion_to_euler(q):
    # Extract components
    w, x, y, z = q

    # Conversion
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z


class LQR:
    # K = np.array([[-1,	-1.5512,	6.9235,	1.5845]])
    # K = np.array([[-0.31622777, 46.18915178, -1.35648665, 13.04275023]])
    # K = np.array([[-1.        , 52.27411189, -2.62866594, 14.99104346]])
    K = np.array([[-31.6227766 , 162.92827674, -30.40599676,  50.15742702]])
    # K = np.array([[1.        , 1.43366053, 3.54176195, 0.64269921]])

    dt = 0.001
    prev_time = time.time()

    state_dbg = np.array([0,0,0,0])

    def compute_state(self, angle_sensor, position_sensor, velocity_sensor, ang_velocity_sensor):
        # For future reference: One of the things that was likely wrong was that the accelerometer to angle computation did not account for the linear accelerations
        # As such, the "angle" that I would compute would be wrong


        # TODO Next time. Verify that this sensor data is correct.
        x = -position_sensor.data[0]
        x_dot = -velocity_sensor.data[0]
        theta = angle_sensor
        theta_dot = ang_velocity_sensor.data[1]


        # For future reference, one of the things that was wrong was that the order of the state vars here vs in the LQR solver was difference...

        return np.array([x, theta, x_dot, theta_dot])
        # return np.array([x, x_dot, theta, theta_dot])
    
    def step(self, angle_sensor, position_sensor, velocity_sensor, ang_velocity_sensor):
        # if time.time() - self.prev_time < self.dt:
        #     return
        
        state = self.compute_state(angle_sensor, position_sensor, velocity_sensor, ang_velocity_sensor)
        self.state_dbg = state
        # u is a force

        # ordered_state = np.array([state[0], state[2], -state[1], state[3]])
        # u = np.matmul(self.K, ordered_state)
        u = np.matmul(self.K, state)

        # convert u to torque, given wheel radius
        r = 0.03
        uT = u*r
        
        # Send half the torque to each wheel
        # uT /= 2.0

        # print("State = ", state)
        # print("K*state = ", self.K * state)
        # print("u: ", uT)
        d.ctrl[:] = uT
        print(uT)
        print(d.ctrl)
        print("------------------------------\n")



m = mujoco.MjModel.from_xml_path('./cart_pole.xml')
d = mujoco.MjData(m)

# print(f"Timestep {m.opt.timestep}")

controller = LQR()

qvels_dbg = np.array([])
ctrls_dbg = np.array([])

x_dbg = np.array([])
theta_dbg = np.array([])
x_dot_dbg = np.array([])
theta_dot_dbg = np.array([])

ctr = 0
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():

        ctr += 1
        step_start = time.time()

        mujoco.mj_step(m, d)

        angle = quaternion_to_euler(d.sensor('angle_sensor').data)[1]
        controller.step(angle, d.sensor('base_position'), d.sensor('base_velocity'), d.sensor('base_ang_velocity'))

        if abs(angle) > 15.0*np.pi/180.0:
            print("15 deg angle exceeded. Terminating...")
            # break

        # if ctr > 1000:
        #     break

        # time.sleep(0.1)

        print(f"Angle: {angle*180/np.pi}")

        qvels_dbg = np.append(qvels_dbg, d.qvel[-1])
        ctrls_dbg = np.append(ctrls_dbg, d.ctrl[0])
        x_dbg = np.append(x_dbg, controller.state_dbg[0])
        theta_dbg = np.append(theta_dbg, controller.state_dbg[1])
        x_dot_dbg = np.append(x_dot_dbg, controller.state_dbg[2])
        theta_dot_dbg = np.append(theta_dot_dbg, controller.state_dbg[3])

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.cam.lookat = d.sensor('base_position').data

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)



# Visualize data at the end of the run
print("Showing plots...")
fig, ax = plt.subplots(2, 3)
stride = 1

ax[0,0].plot(qvels_dbg[::stride])
ax[0,0].legend(["Velocity"])

ax[0,1].plot(ctrls_dbg[::stride])
ax[0,1].legend(["Control"])

ax[0,2].plot(x_dbg[::stride])
ax[0,2].legend(["X"])

ax[1,0].plot(theta_dbg[::stride])
ax[1,0].legend(["Theta"])

ax[1,1].plot(x_dot_dbg[::stride])
ax[1,1].legend(["X_dot"])

ax[1,2].plot(theta_dot_dbg[::stride])
ax[1,2].legend(["Theta_dot"])

plt.show()

print("Done plotting")