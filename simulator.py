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
    # K = np.array([[-31.6227766 , 162.92827674, -30.40599676,  50.15742702]])
    K = np.array([[-100.        ,  346.86699245,  -81.69049517,  108.76436008]])

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

EPISODE_LENGTH_S = 6.0
# print(f"Timestep {m.opt.timestep}")

controller = LQR()

times_dbg = np.array([])
qvels_dbg = np.array([])
ctrls_dbg = np.array([])

x_dbg = np.array([])
theta_dbg = np.array([])
x_dot_dbg = np.array([])
theta_dot_dbg = np.array([])

renderer_fps = 25
simulation_timestep = m.opt.timestep

# Pick the fps that most closely matches a multiple of the simulation timestep. This facilitates the "realtime rendering" logic
renderer_fps = 1/(round((1/renderer_fps)/simulation_timestep) * simulation_timestep)
physics_updates_per_frame = int((1.0/renderer_fps)/simulation_timestep)

print(f"Running simulation at:\n    {renderer_fps} fps \n    {simulation_timestep} physics timestep.")
print(f"    {physics_updates_per_frame} physics updates per rendered frame.")

debug_ctr = 0
with mujoco.viewer.launch_passive(m, d) as viewer:
    prev_iteration_time = 0 # Don't initialize to current time. The delta used for realtime factor might cause a DIV0 error
    sim_start_time = time.monotonic()
    while viewer.is_running() and d.time < EPISODE_LENGTH_S:
        
        # This must be the first thing in the loop. Every single computation from here on will eat into the delta time between frames.
        next_frame_at = time.time_ns() + (1/renderer_fps) * 1_000_000_000

        realtime_factor = ((1/renderer_fps * 1_000_000_000)/(time.time_ns()-prev_iteration_time))
        print(f"{realtime_factor:.2f}X realtime.", end="\r")
        prev_iteration_time = time.time_ns()

        # Perform only as many physics steps are would fit in the time between frames
        # If performing more than this, it would appear as if the simulation were sped up
        for _ in range(physics_updates_per_frame):
            mujoco.mj_step(m, d)
            debug_ctr += 1

            angle = quaternion_to_euler(d.sensor('angle_sensor').data)[1]
            controller.step(angle, d.sensor('base_position'), d.sensor('base_velocity'), d.sensor('base_ang_velocity'))

        # Terminate episode if max angle exceeded
        if abs(angle) > 15.0*np.pi/180.0:
            print("15 deg angle exceeded. Terminating...")
            break

        times_dbg     = np.append(times_dbg, d.time)
        qvels_dbg     = np.append(qvels_dbg, d.qvel[-1])
        ctrls_dbg     = np.append(ctrls_dbg, d.ctrl[0])
        x_dbg         = np.append(x_dbg, controller.state_dbg[0])
        theta_dbg     = np.append(theta_dbg, controller.state_dbg[1])
        x_dot_dbg     = np.append(x_dot_dbg, controller.state_dbg[2])
        theta_dot_dbg = np.append(theta_dot_dbg, controller.state_dbg[3])

        # The last thing that should happen is the rendering. Anything else will eat into the delta time between frames.
        with viewer.lock():
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            # Keep the camera pointed at the base of the robot
            viewer.cam.lookat = d.sensor('base_position').data

            # Render frame
            viewer.sync()

            if time.time_ns() > next_frame_at:
                print("\n\nWARNING: Renderer fps missed. Iteration time too long.")
                continue

            while time.time_ns() < next_frame_at:    
                continue # Busy wait. This is much better for timing than time.sleep(), which is not precise.


sim_end_time = time.monotonic()
print(f"Simulation wallclock time: {sim_end_time - sim_start_time:.3f} s")
print(f"Simulation physics time: {m.opt.timestep * debug_ctr:.3f} s")

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