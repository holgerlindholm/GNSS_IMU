import pandas as pd
from coordinate_converter import gpstime_to_datetime

def read_imu_csv(file_path, gps_week=None):
    # see https://geodesy.noaa.gov/CORS/resources/gpscals.shtml
    try:
        df = pd.read_csv(file_path, sep=r"\s+")
        if gps_week is not None:
            df['datetime'] = df.apply(lambda row: gpstime_to_datetime(gps_week, row['Time']), axis=1)
        return df
        
    except Exception as e:
        print(f"Error reading the IMU CSV file: {e}")
        return None
import pandas as pd

def read_ground_truth_csv(file_path):
    try:
        header = [
            'GPSTime', 'X-ECEF', 'Y-ECEF', 'Z-ECEF',
            'Heading', 'Pitch', 'Roll',
            'VX-ECEF', 'VY-ECEF', 'VZ-ECEF', 'UTCTime'
        ]

        # Read file, skipping header lines
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            skiprows=21,
            names=header,
            engine="python"  # needed for flexible parsing
        )

        # Convert everything to float
        df = df.astype(float)

        return df

    except Exception as e:
        print(f"Error reading the ground truth CSV file: {e}")
        return None

if __name__ == "__main__":
    csv_file = r"data\run2_imu.txt"
    ground_truth_file = r"data\run2_groundtruth.txt"

    imu_data = read_imu_csv(csv_file, gps_week=2415) 
    print(imu_data.head())
    print(imu_data.columns)
    print(imu_data.shape)

    ground_truth_data = read_ground_truth_csv(ground_truth_file)
    print(ground_truth_data.head())
    print(ground_truth_data.columns)
    print(ground_truth_data.shape)