# IMU CSV Reader
import pandas as pd

class IMUData:
    """
    Class to read and store IMU data from a CSV file.
    The CSV file is expected to have the following columns:
    - timestamp: The time at which the IMU data was recorded (100Hz)
    - ax: Acceleration in the x-axis (m/s^2)
    - ay: Acceleration in the y-axis (m/s^2)
    - az: Acceleration in the z-axis (m/s^2)
    - gx: Angular velocity around the x-axis (rad/s)
    - gy: Angular velocity around the y-axis (rad/s)
    - gz: Angular velocity around the z-axis (rad/s)

    """
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.timestamps = self.data['timestamp']

    

    