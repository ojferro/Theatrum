import time

import mujoco
import mujoco.viewer

import numpy as np
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
    K = np.array([[-31.6227766 , 162.92827674, -30.40599676,  50.15742702]])

    dt = 0.001
    prev_time = time.time()

    state_dbg = np.array([0,0,0,0])

    def compute_state(self, angle_sensor, position_sensor, velocity_sensor, ang_velocity_sensor):
        # For future reference: One of the things that was likely wrong was that the accelerometer to angle computation did not account for the linear accelerations
        # As such, the "angle" that I would compute would be wrong

        x = -position_sensor.data[0]
        x_dot = -velocity_sensor.data[0]
        theta = angle_sensor
        theta_dot = ang_velocity_sensor.data[1]


        # For future reference, one of the things that was wrong was that the order of the state vars here vs in the LQR solver was different...

        return np.array([x, theta, x_dot, theta_dot])
        # return np.array([x, x_dot, theta, theta_dot])
    
    def step(self, angle_sensor, position_sensor, velocity_sensor, ang_velocity_sensor):
        # if time.time() - self.prev_time < self.dt:
        #     return
        
        state = self.compute_state(angle_sensor, position_sensor, velocity_sensor, ang_velocity_sensor)
        self.state_dbg = state
        # u is a force
        u = np.matmul(self.K, state)

        # convert u to torque, given wheel radius
        r = 0.03
        uT = u*r
        
        # Send half the torque to each wheel
        d.ctrl[:] = uT/2



m = mujoco.MjModel.from_xml_path('./cart_pole.xml')
d = mujoco.MjData(m)

# print(f"Timestep {m.opt.timestep}")

controller = LQR()

times_dbg = np.array([])
qvels_dbg = np.array([])
ctrls_dbg = np.array([])

x_dbg = np.array([])
theta_dbg = np.array([])
x_dot_dbg = np.array([])
theta_dot_dbg = np.array([])

ctr = 0
step_start = time.time_ns()
with mujoco.viewer.launch_passive(m, d) as viewer:
    sim_start_time = time.time()
    while viewer.is_running() and d.time < 6.0:

        ctr += 1

        mujoco.mj_step(m, d)

        angle = quaternion_to_euler(d.sensor('angle_sensor').data)[1]
        controller.step(angle, d.sensor('base_position'), d.sensor('base_velocity'), d.sensor('base_ang_velocity'))

        if abs(angle) > 15.0*np.pi/180.0:
            print("15 deg angle exceeded. Terminating...")
            # break

        times_dbg = np.append(times_dbg, d.time)
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
        time_until_next_step = m.opt.timestep*(10**9) - (time.time_ns() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step/(10**9))

        step_start = time.time_ns()


sim_end_time = time.time()
print(f"Simulation time: {sim_end_time - sim_start_time}")


# Visualize data at the end of the run
print("Showing plots...")
fig, ax = plt.subplots(2, 3)
stride = 1

ax[0,0].plot(times_dbg[::stride], qvels_dbg[::stride])
ax[0,0].legend(["Velocity"])

ax[0,1].plot(times_dbg[::stride], ctrls_dbg[::stride])
ax[0,1].legend(["Control"])

ax[0,2].plot(times_dbg[::stride], x_dbg[::stride])
ax[0,2].legend(["X"])

ax[1,0].plot(times_dbg[::stride], theta_dbg[::stride])
ax[1,0].legend(["Theta"])

ax[1,1].plot(times_dbg[::stride], x_dot_dbg[::stride])
ax[1,1].legend(["X_dot"])

ax[1,2].plot(times_dbg[::stride], theta_dot_dbg[::stride])
ax[1,2].legend(["Theta_dot"])

plt.show()

print("Done plotting")