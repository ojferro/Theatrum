import numpy as np
class TemporalFilter:
    def __init__(self, alpha=0.9, init_val=0):
        self.alpha = alpha
        self.prev_val = init_val

    def filter(self, val):
        filtered_val = self.alpha * val + (1 - self.alpha) * self.prev_val

        # TODO: See whether previous should be filtered or raw value.
        self.prev_val = val
        return filtered_val
    

class AdaptiveComplementaryFilter:
    ''' Scales the alpha in a complementary filter by the magnitude of the gyro reading.
        * It gives MORE importance to the gyro with large gyro readings, since Coriolis forces introduce
        accelerations on accelerometer data.
        * It gives LESS importance to gyro with small gyro readings, since accelerometer is expeced to be less subject to Coriolis forces.
    '''
    def __init__(self, dt, base_alpha=0.99, alpha_variation_range=0.05, max_gyro_val=100.0):

        self.dt = dt
        self.base_alpha = base_alpha
        self.alpha_variation_range = min(alpha_variation_range, 1-base_alpha)
        print("Using alpha variation range: ", self.alpha_variation_range)
        assert False, "false"

        self.max_gyro_val = max_gyro_val

        self.pitch = 0.0

    def map_to_range(self, val, from_min, from_max, to_min, to_max):

        val = np.clip(val, from_min, from_max)
        
        # Calculate the scaling factor
        scale = (to_max - to_min) / (from_max - from_min)
        # Apply the transformation
        return to_min + (val - from_min) * scale

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

        # Add Gaussian noise to simulate sensor noise
        gyro_y += np.random.normal(0, 0.04)

        # Calculate pitch and roll from accelerometer data
        accel_angle_pitch = self.calculate_angle_from_accelerometer(accel, dynamic_accel_est)

        # Integrate the gyroscope data
        gyro_angle_pitch = gyro_y * self.dt

        min_val = self.base_alpha - self.alpha_variation_range
        max_val = self.base_alpha + self.alpha_variation_range
        alpha = self.map_to_range(np.abs(gyro_y), 0, self.max_gyro_val, min_val, max_val)

        print(f"alpha: {alpha}")

        # Apply complementary filter
        self.pitch = alpha * (self.pitch + gyro_angle_pitch) + (1 - alpha) * accel_angle_pitch
        
        # Negate to match model convention
        return self.pitch
    