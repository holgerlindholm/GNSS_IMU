TODO: 
Deliverables: 
- Make a losely coupled kalman filter for IMU and GNSS integration
    - Simulate GNSS dropout
    - Simulate just IMU data
    - Simulate ground truth data (GNSS)

# Match time stamps between IMU and GNSS data

# Data loading
1. Make csv file reader (IMU data)

2. Import rinex reader

3. Find RINEX data from B328 + precise orbit data (Holger)
    Data of recording: 2026-04-23 --> GPS time Week 2415 (doy 113)
    https://cddis.nasa.gov/archive/gnss/products/2415/ 
    

# Body frame - ECEF coorindate conversion 
# Find IMU biases from stationary data 
# FInd errors in IMU data


# Implement losely coupled kalman filter. 
    - Setup Kalman filter class 

# Plotting functions
- PLot poistion overlayed with sime sort of ortophoto image
- PLot IMU bias over time
- PLot error over time



