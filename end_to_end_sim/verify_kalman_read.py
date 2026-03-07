from read_kalman_shm import KalmanShmReader
import time

reader = KalmanShmReader()

try:
    while True:
        snap = reader.read_snapshot()
        if snap:
            print(snap)
        time.sleep(0.1)
finally:
    reader.close()