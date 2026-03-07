import time
import struct
from multiprocessing import shared_memory, resource_tracker

from end_to_end_sim.read_sensor_shm import BLOCK_SIZE

SHM_NAME = "kalman_shm"
STRUCT_FORMAT = "<IQ7f"
STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)

class KalmanShmWriter:
    def __init__(self, name = SHM_NAME):

        try:
            old_shm = shared_memory.SharedMemory(name=name, create=False)
            old_shm.close()
            old_shm.unlink()
        except FileNotFoundError:
            pass

        self.shm = shared_memory.SharedMemory(name=name, size = STRUCT_SIZE, create=True)
        resource_tracker.unregister(self.shm._name, "shared_memory")
    
        self.buf = self.shm.buf
        self.seq = 0
        self.write_state([0.0]*7, 0)

    def write_state(self, state, timestamp):
        # state = [x, y, dx, dy, d2x, d2y, turn_angle]
        data = struct.pack("<IQ7f", self.seq, timestamp, *state)
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