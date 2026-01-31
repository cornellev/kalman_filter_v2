import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import math

#state is position and velocity
dt = 1 #update every second
F = np.array([[1, dt], [0, 1]]) #state transition matrix based on physics equations
#assume that there is no control matrix B
Q = np.array([[0.1, 0], [0, 0.1]]) #process noise covariance
H = np.identity(2) #measurement matrix to relate sensors and state
R = np.array([[0.15, 0], [0, 0.15]]) #measurement noise covariance
x_k = [[0, 1]] #true values
z_k = [] #sensor measurements
for i in range(1, 11):
    x_k.append(F @ x_k[i-1] + np.random.multivariate_normal([0, 0], Q))
for i in range(0, 11):
    z_k.append(H @ x_k[i] + np.random.multivariate_normal([0, 0], R))
x_k = np.array(x_k)
z_k = np.array(z_k)
x_hat_0 = [0, 1] #initial state estimate
P_0 = np.array([[0.1, 0], [0, 0.1]]) #initial covariance
x_hat_k = [x_hat_0] #predictions array
P_k = [P_0] #covariance array
x_hat_prev = x_hat_0
P_prev = P_0

for i in range(1, 11):
    x_hat_curr_minus = F @ np.array(x_hat_prev).T
    P_minus = F @ np.array(P_prev) @ F.T + Q
    z_hat_curr = H @ x_hat_curr_minus
    residual = z_k[i].T - z_hat_curr
    S = H @ P_minus @ H.T + R
    K = P_minus @ H.T @ np.linalg.inv(S)
    x_hat_new = x_hat_curr_minus + K @ residual
    P_new = (np.identity(2) - K @ H) @ P_minus @ (np.identity(2) - K @ H).T + K @ R @ K.T

    print("Predicted state at time step", i)
    print(x_hat_new)
    print()

    x_hat_prev = x_hat_new
    P_prev = P_new

    x_hat_k.append(x_hat_new)
    P_k.append(P_new)

x_hat_k = np.array(x_hat_k)
P_k = np.array(P_k)
t = np.arange(len(x_k))

rmse = math.sqrt(mse(x_k, x_hat_k))
print("Overall RMSE: ", rmse)

true_pos = x_k[:, 0]
sensor_pos = z_k[:, 0]
pred_pos = x_hat_k[:, 0]

true_vel = x_k[:, 1]
sensor_vel = z_k[:, 1]
pred_vel = x_hat_k[:, 1]

plt.figure()
plt.plot(t, true_pos, label="True Position")
plt.scatter(t, sensor_pos, label="Sensor Measurements")
plt.plot(t, pred_pos, label="KF Prediction")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.title("Kalman Filter Position Tracking")
plt.legend()
plt.show()