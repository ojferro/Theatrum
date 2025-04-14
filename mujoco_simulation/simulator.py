import time

import mujoco
import mujoco.viewer

import numpy as np
import matplotlib.pyplot as plt

from filters import TemporalFilter

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

    K = np.array([[3.16227766, 31.53989481,  3.65425785,  4.92520868]])
    

    dt = 0.001
    prev_time = time.time()

    state_dbg = np.array([0,0,0,0])

    position_setpoint = 0.0
    velocity_setpoint = 0.0

    def __init__(self, dt):
        self.dt = dt
        self.prev_time = time.time()

    def compute_state(self, angle_sensor, position_sensor, velocity_sensor, ang_velocity_sensor):
        # For future reference: One of the things that was likely wrong was that the accelerometer to angle computation did not account for the linear accelerations
        # As such, the "angle" that I would compute would be wrong

        x = position_sensor
        x_dot = velocity_sensor
        theta = angle_sensor
        theta_dot = ang_velocity_sensor


        # For future reference, one of the things that was wrong was that the order of the state vars here vs in the LQR solver was different...

        return np.array([x, theta, x_dot, theta_dot]) - np.array([self.position_setpoint, 0, self.velocity_setpoint, 0])
    
    def step(self, angle_sensor, position_sensor, velocity_sensor, ang_velocity_sensor):
        
        state = self.compute_state(angle_sensor, position_sensor, velocity_sensor, ang_velocity_sensor)
        # self.state_dbg = state
        # u is a force
        u = np.matmul(self.K, state)

        # convert u to torque, given wheel radius
        r = 0.03
        uT = u*r
        
        # Send half the torque to each wheel
        d.ctrl[:] = uT/2

        # Save state for debugging
        self.state_dbg = state

class ComplementaryFilter:
    def __init__(self, dt, alpha=0.999):

        self.dt = dt
        self.alpha = alpha

        self.pitch = 0.0

    def calculate_angle_from_accelerometer(self, accel_data, dynamic_accel_est):
        # Extract components
        x, y, z = accel_data

        # Conversion
        x -= dynamic_accel_est[0]
        z -= dynamic_accel_est[1]

        # Negate to match model convention
        angle = -np.arctan2(x, z)
        return angle
    
    def step(self, gyro, accel, dynamic_accel_est = np.array([0,0])):
        # Get sensor data
        _gyro_x, gyro_y, _gyro_z = gyro

        # Calculate pitch and roll from accelerometer data
        accel_angle_pitch = self.calculate_angle_from_accelerometer(accel, dynamic_accel_est)

        # Integrate the gyroscope data
        gyro_angle_pitch = gyro_y * self.dt

        # Apply complementary filter
        self.pitch = self.alpha * (self.pitch + gyro_angle_pitch) + (1 - self.alpha) * accel_angle_pitch
        
        # Negate to match model convention
        return self.pitch

class Model:
    def __init__(self, dt):
        m_c = 1.105 - 0.3
        m_p = 0.3
        g = 9.81
        L = 0.3

        # Damping coeffs
        d1 = 0.001
        d2 = 0.001

        self.delta_t = dt

        self.A = np.array([[0,0,1,0],
                [0,0,0,1],
                [0,g*m_p/m_c, -d1/m_c, -d2/(L*m_c)],
                [0, g*(m_c+m_p)/(L*m_c), -d1/(L*m_c), -d2*(m_c+m_p)/(L**2 *m_c * m_p)]
                ])
        self.B = np.array([[0],[0], [1/m_c], [1/(L*m_c)]])

        self.x_ddot = 0.0
        # self.theta_ddot = 0.0

    def step(self, state, u):
        new_state = state + (self.A @ state + self.B @ u) * self.delta_t

        self.x_ddot = (new_state[2] - state[2])/self.delta_t

        print(f"x_ddot: {self.x_ddot}")

        return new_state
    
    def calculate_dynamic_acceleration(self, state):
        ''' Calculate the component of acceleration not due to gravity at the accelerometer location, 0.37m from the pivot.'''
        x = state[0]
        theta = state[1]
        x_dot = state[2]
        theta_dot = state[3]

        L = 0.37  # Distance from pivot point to accelerometer, meters

        tangentialAccel = self.x_ddot*np.cos(theta) #- theta_dot**2*np.cos(theta)
        radialAccel = -self.x_ddot*np.sin(theta) - theta_dot**2 * L

        # P_x_ddot = self.x_ddot - L * (self.theta_ddot * np.cos(theta) - theta_dot**2 * np.sin(theta))
        # P_z_ddot = 0 + L * (self.theta_ddot * np.sin(theta) + theta_dot**2 * np.cos(theta))

        print(f"tangentialAccel: {tangentialAccel}, radialAccel: {radialAccel}")
        print(f"theta_dot from dynamic accel computation: {theta_dot}")

        return np.array([tangentialAccel, radialAccel])


m = mujoco.MjModel.from_xml_path('./assets/wheeled_biped_servos.xml')
d = mujoco.MjData(m)

EPISODE_LENGTH_S = 1000.0
VIZ_IN_REALTIME = True

times_dbg = np.array([])
accel_theta_dot_dbg = np.array([])
filtered_accel_theta_dot_dbg = np.array([])
ctrls_dbg = np.array([])

x_dbg = np.array([])
theta_dbg = np.array([])
accel_theta_dbg = np.array([])
x_dot_dbg = np.array([])
theta_dot_dbg = np.array([])

model_x_dbg = np.array([])
model_theta_dbg = np.array([])
model_x_dot_dbg = np.array([])
model_theta_dot_dbg = np.array([])

renderer_fps = 25
simulation_timestep = m.opt.timestep

controller = LQR(simulation_timestep)
comp_filter = ComplementaryFilter(simulation_timestep)
temporal_filter = TemporalFilter(alpha = 0.5)
# comp_filter = AdaptiveComplementaryFilter(simulation_timestep, base_alpha=0.99)
math_model = Model(simulation_timestep)

# model_predicted_state = np.array([0,0,0,0])

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
        if VIZ_IN_REALTIME:
            next_frame_at = time.time_ns() + (1/renderer_fps) * 1_000_000_000

            realtime_factor = ((1/renderer_fps * 1_000_000_000)/(time.time_ns()-prev_iteration_time))
            print(f"{realtime_factor:.2f}X realtime.", end="\r")
            prev_iteration_time = time.time_ns()

        # Perform only as many physics steps are would fit in the time between frames
        # If performing more than this, it would appear as if the simulation were sped up
        for _ in range(physics_updates_per_frame):
            mujoco.mj_step(m, d)
            debug_ctr += 1

            # accel_angle = calculate_angle_from_accelerometer(d.sensor('accelerometer').data)
            angle = quaternion_to_euler(d.sensor('angle_sensor').data)[1]


            # model_predicted_state = math_model.step(controller.state_dbg, [d.ctrl[0]])
            # model_predicted_state = math_model.step(model_predicted_state, [d.ctrl[0]])
            # model_predicted_state[3] = d.sensor('gyro').data[1]
            # dynamic_accel = math_model.calculate_dynamic_acceleration(model_predicted_state)
            # print(f"GYRO: {d.sensor('gyro').data}")

            # Simulate sensor noise
            gyro = d.sensor('gyro').data
            gyro += np.random.normal(0, 0.04*1.5, size=(3,))
            accel = d.sensor('accelerometer').data
            accel += np.random.normal(0, 0.14*1.5, size=(3,))

            cf_angle = comp_filter.step(gyro, accel) #dynamic_accel)
            thetadot = temporal_filter.filter(gyro[1])

            wheel_angular_position = d.sensor('wheel_pos').data[0]
            wheel_angular_velocity = d.sensor('wheel_vel').data[0]  + np.random.normal(0, 0.01)
            x = wheel_angular_position * 0.03
            xdot = wheel_angular_velocity * 0.03

            # Use GT readings
            # controller.step(angle, d.sensor('base_position'), d.sensor('base_velocity'), d.sensor('base_ang_velocity'))

            # Use estimated readings
            controller.step(cf_angle, x, xdot, thetadot)


        # Terminate episode if max angle exceeded
        if abs(angle) > 80.0*np.pi/180.0:
            print("80 deg angle exceeded. Terminating...")
            break

        times_dbg                    = np.append(times_dbg, d.time)
        ctrls_dbg                    = np.append(ctrls_dbg, d.ctrl[0])
        x_dbg                        = np.append(x_dbg, d.sensor('base_position').data[0])
        theta_dbg                    = np.append(theta_dbg, angle * 180/np.pi)
        accel_theta_dbg              = np.append(accel_theta_dbg, cf_angle * 180/np.pi)
        x_dot_dbg                    = np.append(x_dot_dbg, d.sensor('base_velocity').data[0])
        theta_dot_dbg                = np.append(theta_dot_dbg, d.sensor('base_ang_velocity').data[1] * 180/np.pi)
        accel_theta_dot_dbg          = np.append(accel_theta_dot_dbg, gyro[1] * 180/np.pi)
        filtered_accel_theta_dot_dbg = np.append(filtered_accel_theta_dot_dbg, thetadot * 180/np.pi)

        # model_x_dbg         = np.append(model_x_dbg,         model_predicted_state[0])
        # model_theta_dbg     = np.append(model_theta_dbg,     model_predicted_state[1])
        # model_x_dot_dbg     = np.append(model_x_dot_dbg,     model_predicted_state[2])
        # model_theta_dot_dbg = np.append(model_theta_dot_dbg, model_predicted_state[3])

        # The last thing that should happen is the rendering. Anything else will eat into the delta time between frames.
        with viewer.lock():
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            # Keep the camera pointed at the base of the robot
            viewer.cam.azimuth= -90.0
            viewer.cam.lookat = d.sensor('base_position').data

            # Render frame
            viewer.sync()

            if VIZ_IN_REALTIME:
                if time.time_ns() > next_frame_at:
                    print("\n\nWARNING: Renderer fps missed. Iteration time too long.")
                    continue

                while time.time_ns() < next_frame_at:    
                    continue # Busy wait. This is much better for timing than time.sleep(), which is not precise.


sim_end_time = time.monotonic()
print(f"Simulation wallclock time: {sim_end_time - sim_start_time:.3f} s")
print(f"Simulation physics time: {m.opt.timestep * debug_ctr:.3f} s")

# Visualize data at the end of the run

if False:
    print("Showing plots...")
    fig, ax = plt.subplots(2, 3)
    stride = 1

    # Plot the data

    ### X ###
    ax[0,0].plot(times_dbg[::stride], x_dbg[::stride])
    ax[0,0].legend(["X"])

    ### X_dot ###
    ax[1,0].plot(times_dbg[::stride], x_dot_dbg[::stride])
    ax[1,0].legend(["X_dot"])

    ### Theta ###
    # GT Theta
    ax[0,1].plot(times_dbg[::stride], theta_dbg[::stride])

    # Complementary filter Theta
    ax[0,1].plot(times_dbg[::stride], accel_theta_dbg[::stride], linestyle="--")
    ax[0,1].legend(["Theta", "IMU Theta"])


    ### Theta_dot ###
    ax[1,1].plot(times_dbg[::stride], theta_dot_dbg[::stride])
    ax[1,1].plot(times_dbg[::stride], accel_theta_dot_dbg[::stride])
    ax[1,1].plot(times_dbg[::stride], filtered_accel_theta_dot_dbg[::stride], linestyle="--")
    ax[1,1].legend(["Theta_dot", "IMU Theta_dot", "Filtered IMU Theta_dot"])

    ### Control ###
    ax[0,2].plot(times_dbg[::stride], ctrls_dbg[::stride])
    ax[0,2].legend(["Control"])

    ### q_vels ###
    ax[1,2].plot(times_dbg[::stride], accel_theta_dot_dbg[::stride])
    ax[1,2].legend(["Est theta_dot"])


    plt.show()

    print("Done plotting")