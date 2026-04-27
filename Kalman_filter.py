import numpy as np
from coordinate_converter import (
    attitude_update,
    velocity_update,
    position_update,
    skew_symmetric
)

class KF: #not a standard direct state KF but a error state KF
    def __init__(self, r0_e, v0_e, C_b_e0, P, Q, R, g_e):

        #nominal state (best physical estimate) expanded rather than state vetcor
        self.r_e = r0_e #position ECEF (x,y,z)
        self.v_e = v0_e #velocitiy ECEF (vx,vy,vz)
        self.C_b_e = C_b_e0 #attitude (rotation matrix from body to ECEF)
        self.b_a = np.zeros(3) #accel bias (ax,ay,az)
        self.b_g = np.zeros(3) #gyro bias (pitch, roll, yaw)

        self.P = P #covariance (15,15)
        self.Q = Q #procces noise (15,15)
        self.R = R #measurment noise (3,3) #x,y,z in gnss
        self.g_e = g_e #gravity vector in ECEF frame

    def predict(self, accel_b_meas, gyro_b_meas, dt):

        #since the KF estimates error state (direct estimation is poor due to linearization issues with attitude
        #and large values for position)
        #the estimates state is below:
        #del_x = [del_r (position error),del_v (velocity error),del_theta (attitude error),
        #         delb_a (accel bias error), delb_g (gyro bias error)]

        #remove filters estimated biases from IMU measurements
        #accel_b_meas and gyro_b_meas are IMU gyro and accel in body frame
        a_b = accel_b_meas - self.b_a
        omega_b = gyro_b_meas - self.b_g

        #normal INS mechanisation/propogation
        #save old INS states (t-1)
        C_old = self.C_b_e #old attitude from body to ECEF rotation
        v_old = self.v_e #old velocity
        r_old = self.r_e #old position

        #propgate attitude using helper from coord converter
        #gyro angular rate integrated to give angles
        C_new = attitude_update(C_old, omega_b, dt)

        #propogate velocity using helper
        #handles conversion to ECEF from body + propogation with accel and timestep dt
        #there is also navigation corrections included (see helepr):
        # specific force + gravity - Earth rotation/Coriolis terms
        v_new, f_e = velocity_update(v_old, C_old, C_new, a_b, self.g_e, dt)
        #v_new, f_e = velocity_update(v_old, C_old, C_new, a_b, np.zeros(3), dt)

        #propogate postion using helper
        #propogate with velocity and dt
        #there is also navigation coorections included here (see helper):
        #specific force + gravity - Earth rotation/Coriolis terms
        r_new = position_update(r_old, f_e, v_new, v_old, self.g_e, dt)
        #r_new = position_update(r_old, f_e, v_new, v_old, np.zeros(3), dt)

        #store new estimates for propogation
        self.C_b_e = C_new
        self.v_e = v_new
        self.r_e = r_new

        #covariance prediction

        #create state transition matrix (simplified)
        #see Daniel's PhD thesis eq 5.2 for the full version
        omega_ie = 7.29211585494e-5
        Omega_ie_e = skew_symmetric(np.array([0.0, 0.0, omega_ie]))

        F = np.eye(15)

        # Position error dynamics
        F[0:3, 0:3] += -Omega_ie_e * dt #pose velocity
        F[0:3, 3:6] = np.eye(3) * dt #attitude velocity coupling

        # Velocity error dynamics
        F[3:6, 3:6] += -2.0 * Omega_ie_e * dt

        # attitude error -> velocity error
        F[3:6, 6:9] = -skew_symmetric(f_e) * dt

        # accel bias -> velocity error
        F[3:6, 9:12] = -self.C_b_e * dt

        # Attitude error dynamics
        F[6:9, 6:9] += -Omega_ie_e * dt

        # gyro bias -> attitude error
        F[6:9, 12:15] = -self.C_b_e * dt

        # Biases stay random-walk / constant
        F[9:12, 9:12] = np.eye(3)
        F[12:15, 12:15] = np.eye(3)

        #propoate covarinace with general formula
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_pos_ecef, z_vel_ecef):



        #create observation matrix
        H = np.zeros((6, 15))
        # position observes r
        H[0:3, 0:3] = np.eye(3)

        # velocity observes v
        H[3:6, 3:6] = np.eye(3)

        z_pred = np.hstack((self.r_e, self.v_e))
        z = np.hstack((z_pos_ecef, z_vel_ecef))

        #compute innovation (dont need to do H@X since we seperate state vector already)
        y = z - z_pred

        #compute innovation covariance
        S = H @ self.P @ H.T + self.R  #(6,6)

        #compute kalman gain
        K = self.P @ H.T @ np.linalg.inv(S) #(15,3)

        #instead of update state directly we first compute corrections to INS (K@y)
        #then we apply the corrections individually to each state term
        dx = K @ y #(15,1)

        #dx[9:15] = 0.0

        #state update per state value
        self.r_e = self.r_e + dx[0:3] #first 3 terms correspond to x,y,z in our state rep and so on for rest of the terms
        self.v_e = self.v_e + dx[3:6] #velcoity correction

        dtheta = dx[6:9] #angular corrections from dx
        #cant add correction directly to rot matrix
        #so create small angle correct = I - skew_mat(dtehta)
        #then apply that to rot matrix: small angle correct @ rot matrix
        self.C_b_e = (np.eye(3) + skew_symmetric(dtheta)) @ self.C_b_e #attitude correction
        #self.C_b_e = self.C_b_e @ (np.eye(3) - skew_symmetric(dtheta))

        #update biases
        self.b_a = self.b_a + dx[9:12]
        self.b_g = self.b_g + dx[12:15]

        #covariance update with joseph form for numerical stability
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T

        return y, S