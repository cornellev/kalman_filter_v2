import time
import struct
from multiprocessing import shared_memory, resource_tracker
SHM_NAME = "kalman_shm"

KALMAN_FMT = "<Q7f"
KALMAN_SIZE = struct.calcsize(KALMAN_FMT)

SEQ_FMT = "<I"
SEQ_SIZE = struct.calcsize(SEQ_FMT)
BLOCK_SIZE = SEQ_SIZE + KALMAN_SIZE

def _read_seq(buf) -> int:
    return struct.unpack_from(SEQ_FMT, buf, 0)[0]

class KalmanShmReader:
    def __init__(self, name: str = SHM_NAME):
        self.available = False
        self._shm = None
        self._buf = None

        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            resource_tracker.unregister(shm._name, "shared_memory")
        except FileNotFoundError:
            print("Kalman SHM not found.")
            return

        if shm.size < BLOCK_SIZE:
            shm.close()
            raise RuntimeError(f"SHM too small: {shm.size} < {BLOCK_SIZE}")

        self._shm = shm
        self._buf = shm.buf
        self.available = True

    def close(self):
        """Detach from the shared memory block."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
            self._buf = None
            self.available = False

    def read_snapshot(self):
        """
        Read a single consistent snapshot using a seq-lock protocol.
        Returns (seq:int, data:tuple) or None if unavailable.
        """
        if not self.available:
            return None

        buf = self._buf
        while True:
            seq1 = _read_seq(buf)
            if seq1 & 1:
                continue

            data = struct.unpack_from(KALMAN_FMT, buf, SEQ_SIZE)

            seq2 = _read_seq(buf)
            if seq1 == seq2 and not (seq2 & 1):
                return seq2, data

    def read_snapshot_dict(self):
        """
        Read a snapshot and return it as a structured dict, or None if unavailable.
        """
        snap = self.read_snapshot()
        if snap is None:
            return None

        seq, d = snap
        return {
            "seq": seq,
            "global_ts": d[0],
            "x": d[1],
            "y": d[2],
            "dx": d[3],
            "dy": d[4],
            "d2x": d[5],
            "d2y": d[6],
            "turn_angle": d[7],
        }

def main():
    RATE = 10
    PERIOD = 1 / RATE

    reader = KalmanShmReader()
    if not reader.available:
        return 1

    try:
        while True:
            snap = reader.read_snapshot_dict()
            if snap is not None:
                print(snap)
            time.sleep(PERIOD)
    finally:
        reader.close()

if __name__ == "__main__":
    raise SystemExit(main())