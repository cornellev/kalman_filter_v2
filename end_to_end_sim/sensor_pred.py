from end_to_end_sim.read_sensor_shm import SensorShmReader
import ekf
import physics as sim
import numpy as np
from end_to_end_sim.write_kalman_shm import KalmanShmWriter
import time

reader = SensorShmReader()
writer = KalmanShmWriter()

try:
    # state = [x, y, dx, dy, d2x, d2y, turn_angle]
    state_dim = 7
    sensor_dim = 6
    state = [0, 0, 0, 0, 0, 0, 0]
    Q = np.eye(state_dim) * 0.01
    R = np.eye(sensor_dim) * 0.15
    while True:
        snap = reader.read_snapshot()
        if snap:
            # u = [throttle, v, turn_angle]
            # z = [x, y, rpm_fl, rpm_fr, rpm_rl, rpm_rr]
            u = [snap[1][18], snap[1][17], snap[1][6]]
            z = [snap[1][14], snap[1][15], snap[1][8], snap[1][9], snap[1][11], snap[1][12]]
            state, P = ekf.ekf(state, u, P, z, Q, R, sim.f, sim.F_jacobian, sim.H_matrix, sim.h)
            writer.write_state(state.tolist(), snap[1][0])
            print("Estimated State:", state)
        time.sleep(1)
finally:
    reader.close()
    writer.close()