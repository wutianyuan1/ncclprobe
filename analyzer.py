from multiprocessing import shared_memory, resource_tracker
import numpy as np

METADATA_FIELDS = 5
SIZEOF_INT64 = 8


def remove_shm_from_resource_tracker():
    """This is a bug of multiprocessing.shared_memory, manully fix it here
    Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked
    More details at: https://bugs.python.org/issue38119
    """
    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


class NcclRecord(object):
    def __init__(self, num_fields, max_records):
        remove_shm_from_resource_tracker()
        self.shm_size = (num_fields * max_records + METADATA_FIELDS) * SIZEOF_INT64
        self.shm = shared_memory.SharedMemory(
            "ncclRecord", create=False, size=self.shm_size)
        self.data = np.frombuffer(self.shm.buf, np.int64)
        self.buffer = self.data[METADATA_FIELDS:]
        self.num_fields = self.data[0]
        self.max_records = self.data[1]

    @property
    def num_records(self):
        return self.data[2]
    
    def __getitem__(self, idx):
        head = self.data[3]
        start = ((head + idx) % self.max_records) * self.num_fields
        end = start + self.num_fields
        return self.buffer[start: end]

    def __str__(self):
        return 'RingBuffer[' + ','.join(str(self[i]) for i in range(self.num_records)) + ']'

    def __repr__(self):
        return str(self)


if __name__ == '__main__':
    rec = NcclRecord(5, 10)
    print(rec, rec[3], sep='\n')
