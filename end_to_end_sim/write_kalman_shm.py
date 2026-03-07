import time
import struct
from multiprocessing import shared_memory, resource_tracker

from end_to_end_sim.read_sensor_shm import BLOCK_SIZE

SHM_NAME = "kalman_shm"

# SENSOR_FMT = "<" + (
#     "Q" +                    # global ts
#     "I" + "f" + "f" +        # power
#     "I" + "f" + "f" +        # steering
#     "I" + "f" + "f" +        # rpm_front
#     "I" + "f" + "f" +        # rpm_back
#     "I" + "f" + "f" +        # gps
#     "I" + "f" + "f"          # motor
# )
# SENSOR_SIZE = struct.calcsize(SENSOR_FMT)

# SEQ_FMT = "<I"
# SEQ_SIZE = struct.calcsize(SEQ_FMT)
# BLOCK_SIZE = SEQ_SIZE + SENSOR_SIZE

# def _read_seq(buf) -> int:
#     return struct.unpack_from(SEQ_FMT, buf, 0)[0]

class KalmanShmWriter:
    def __init__(self, name = SHM_NAME):
        try:
            shm = shared_memory.SharedMemory(name=name, create=True)
            resource_tracker.unregister(shm._name, "shared_memory")
        except FileNotFoundError:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
            resource_tracker.unregister(self.shm._name, "shared_memory")
    
        self.buf = self.shm.buf
        self.seq = 0
        self.write_state([0.0]*7, 0)

    def write_state(self, state, timestamp):
        # state = [x, y, dx, dy, d2x, d2y, turn_angle]
        data = struct.pack("<I7f", self.seq, timestamp, *state)
        self.buf[:len(data)] = data
        self.seq += 1
        struct.pack_into("<I", self.buf, 0, self.seq)

    def close(self):
        self.shm.close()
        
    def unlink(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass