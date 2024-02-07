import time

import mujoco
import mujoco.viewer

import numpy as np
import math

import pdb

np.set_printoptions(suppress=True)

m = mujoco.MjModel.from_xml_path('./cart_pole.xml')
d = mujoco.MjData(m)
# d.qvel[:] = [0, 0, 0, 0, 0, 0, 0.1, 0]


class PID:

    kp = 3
    ki = 12.5
    kd = 0

    prev_error = 0
    pitch_setpoint = 0

    integral = 0

    dt = 0.001
    prev_time = time.time()

    def step(self, accelerometer):
        if time.time() - self.prev_time < self.dt:
            return

        accel_x = accelerometer.data[0]
        accel_y = accelerometer.data[1]
        accel_z = accelerometer.data[2]

        # Calculate the pitch angle
        pitch = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2))

        self.error = pitch - self.pitch_setpoint

        proportional = self.error
        derivative = (self.error - self.prev_error)/self.dt
        self.integral += self.error * self.dt

        u = self.kp * proportional + self.kd * derivative + self.ki * self.integral
        d.ctrl[:] = u

        print("u: ", u)
        print("error: ", self.error)

        self.error = self.error
        self.prev_time = time.time()

class LQR:

    # K = np.array([[  -0.81649658 , -1.87880076 , 28.97246611 , 4.57440451]])
    # K = np.array([[  -0.21649658 , -0.07880076 , 28.97246611 , 4.57440451]])
    # K = np.array([[-0.81649658,  -1.87784817, -28.95096846,  -4.57269212]])
    # K = np.array([[ -0.000710678,  -0.0046742923, 5.87350048, 2.5337124 ]])


    # Wheeled pendulum model
    # K = np.array([[ 1.        , 50.70750281,  1.84997241, 14.61065534]])
    # K = np.array([[ 0.31622777, 31.37509862,  0.64850033,  9.19869613]])
    # K = np.array([[ 1.        , 33.40453971,  1.37608875,  4.62799863]])
    
    # K = np.array([[ -3.1623, -6.0668, 50.3331, 9.2099]])
    K = np.array([[-1,	-1.5512,	6.9235,	1.5845]])
    

    dt = 0.001
    prev_time = time.time()

    def compute_state(self, accelerometer, position_sensor, velocity_sensor, ang_velocity_sensor):
        accel_x = accelerometer.data[0]
        accel_y = accelerometer.data[1]
        accel_z = accelerometer.data[2]

        # Calculate the pitch angle
        pitch = math.atan2(accel_x, accel_z)

        x = position_sensor.data[0]
        x_dot = velocity_sensor.data[0]
        theta = -pitch
        theta_dot = ang_velocity_sensor.data[1]

        return np.array([x, x_dot, theta, theta_dot])
    
    def step(self, accelerometer, position_sensor, velocity_sensor, ang_velocity_sensor):
        if time.time() - self.prev_time < self.dt:
            return
        
        state = self.compute_state(accelerometer, position_sensor, velocity_sensor, ang_velocity_sensor)


        # u is a force
        u = np.matmul(self.K, state)

        # convert u to torque, given wheel radius
        r = 0.03
        uT = u*r # Send half the torque to each wheel


        print("State = ", state)
        print("K*state = ", self.K * state)
        print("u: ", uT)
        d.ctrl[:] = uT

    

controller = LQR()
# controller = PID()

ctr = 0
with mujoco.viewer.launch_passive(m, d) as viewer:
  while viewer.is_running():
    step_start = time.time()

    ctr += 1
    if ctr > 100:
      controller.step(d.sensor('accelerometer'), d.sensor('base_position'), d.sensor('base_velocity'), d.sensor('base_ang_velocity'))
    mujoco.mj_step(m, d)
    # time.sleep(0.01)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
    #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
      # viewer.opt.flags[mujoco.mjtVisFlag] = 1
      
      # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
      viewer.cam.lookat = d.sensor('base_position').data

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
