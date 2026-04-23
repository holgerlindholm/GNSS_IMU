# Coordinate utils
import numpy as np

# def gravity_wgs84(lat_deg, h):
#     # Convert latitude to radians
#     phi = np.radians(lat_deg)

#     # Constants (WGS84)
#     a = 6378137.0  # semi-major axis (m)
#     gamma_e = 9.7803253359  # equatorial gravity (m/s^2)
#     k = 0.00193185265241
#     e2 = 0.00669437999013

#     # Normal gravity at sea level
#     sin_phi = np.sin(phi)
#     gamma = gamma_e * (1 + k * sin_phi**2) / np.sqrt(1 - e2 * sin_phi**2)

#     # Height correction
#     g = gamma * (1 - (2 * h / a) + (3 * (h**2) / a**2))

#     return g

# approximate_pos = {"lat": 55.783270, "lon": 12.517436, "h":49.0}
# print("Gravity at approximate position:", gravity_wgs84(approximate_pos["lat"], approximate_pos["h"]), "m/s^2")
# g = gravity_wgs84(approximate_pos["lat"], approximate_pos["h"]) #  9.81558596948718 m/s^2
# g = np.array([0.0, 0.0, -g]) # Gravity vector in ECEF frame (m/s^2)

def skew_symmetric(vec):
    """
    Convert a 3D vector to a skew-symmetric matrix.
    
    Parameters:
    vec: A 3D vector (numpy array of shape (3,)).
    
    Returns:
    A skew-symmetric matrix corresponding to the input vector.
    """
    return np.array([[0, -vec[2], vec[1]], 
                     [vec[2], 0, -vec[0]], 
                     [-vec[1], vec[0], 0]])

# CONSTANTS
g = 9.81558596948718 # m/s^2 # Gravity at approximate position (55.783270, 12.517436, 49.0) using WGS84 model
g = np.array([0.0, 0.0, -g]) # Gravity vector in ECEF frame (m/s^2)
tau = 0.01 # IMU sampling time (100 Hz)
OMEGA_IE  = 7.29211585494e-5 # Earth rotation rate (rad/s)
Omega_ie_e = skew_symmetric(np.array([0, 0, OMEGA_IE])) # Skew symmetric matrix for Earth rotation 

def attitude_update(C_b_e_minus, omega_ib_b, tau):
    """
    Update the attitude (rotation matrix) using the IMU data and Earth rotation.
    from Daniels PhD thesis (page 93 ((4.6)))
    C_b^{e+} = C_b^{e-} * (I_3 + Omega_{ib^b} * tau) - Omega_{ie^e} * C_b^{e-} * tau

    Parameters:
    C_b_e_minus: The previous attitude (rotation matrix) from body frame to ECEF frame.
    omega_ib_b: The angular rates from the gyroscope (rad/s).
    tau: The sampling time (s).

    Returns:
    C_b_e_plus: The updated attitude (rotation matrix) from body frame to ECEF frame.
    """
    I_3 = np.eye(3) # Identity matrix
    Omega_ib_b = skew_symmetric(omega_ib_b)
    C_b_e_plus = C_b_e_minus @ (I_3 + Omega_ib_b * tau) - Omega_ie_e @ C_b_e_minus * tau
    return C_b_e_plus

def velocity_update(v_eb_e_minus, C_b_e_minus, C_b_e_plus, f_ib_b, tau):
    """
    Velocity update using the IMU data and Earth rotation.
    from Daniels PhD thesis (page 94 ((4.7) - (4.8)))
    v_eb_e+ = v_eb_e- + (f_ib_e + g- 2 * Omega_ie_e * v_eb_e-) * tau
    f_ib_e = 0.5 * (C_b_e- + C_b_e+) * f_ib_b
    Parameters:
    v_eb_e_minus: The previous velocity of the body frame with respect to the ECEF frame, expressed in the ECEF frame.
    C_b_e_minus: The previous attitude (rotation matrix) from body frame to ECEF frame.
    C_b_e_plus: The updated attitude (rotation matrix) from body frame to ECEF frame.
    f_ib_b: The specific force from the accelerometer (m/s^2).
    tau: The sampling time (s).

    Returns:
    v_eb_e_plus: The updated velocity of the body frame with respect to the ECEF frame, expressed in the ECEF frame.
    """

    f_ib_e = 0.5 * (C_b_e_minus + C_b_e_plus) @ f_ib_b
    v_eb_e_plus = v_eb_e_minus + (f_ib_e + g - 2 * Omega_ie_e @ v_eb_e_minus) * tau
    return v_eb_e_plus, f_ib_e

def position_update(r_eb_minus, f_ib_e, v_eb_e_plus, v_eb_e_minus, tau):
    """
    Position update using the velocity in the ECEF frame.
    from Daniels PhD thesis (page 94 ((4.9)))
    r_eb+ = r_eb- + v_eb_e+ * tau + (f_ib_e + g - 2 * Omega_ie_e * v_eb_e-) * tau / 2

    Returns:
    r_eb_plus: The updated position of the body frame with respect to the ECEF frame, expressed in the ECEF frame.
    """
    r_eb_plus = r_eb_minus + v_eb_e_plus * tau + (f_ib_e + g - 2 * Omega_ie_e @ v_eb_e_minus) * (tau) / 2
    return r_eb_plus

def main():
    # Example usage
    C_b_e_minus = np.eye(3) # Initial attitude (identity matrix)
    v_eb_e_minus = np.zeros(3) # Initial velocity (zero)
    r_eb_minus = np.zeros(3) # Initial position (zero)

    # IMU data - TODO: Load actual IMU data from CSV
    omega_ib_b = np.array([0.0, 0.0, 0.0])
    f_ib_b = np.array([0.0, 0.0, 0.0])

    C_b_e_plus = attitude_update(C_b_e_minus, omega_ib_b, tau)
    v_eb_e_plus, f_ib_e = velocity_update(v_eb_e_minus, C_b_e_minus, C_b_e_plus, f_ib_b, tau)
    r_eb_plus = position_update(r_eb_minus, f_ib_e, v_eb_e_plus, v_eb_e_minus, tau)

if __name__ == "__main__":
    main()