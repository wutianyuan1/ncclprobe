from multiprocessing import shared_memory, resource_tracker
import numpy as np
import matplotlib.pyplot as plt

METADATA_FIELDS = 6
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
        if idx >= self.num_records:
            raise StopIteration
        head = self.data[3]
        start = ((head + idx) % self.max_records) * self.num_fields
        end = start + self.num_fields
        return self.buffer[start: end]

    def __str__(self):
        return 'RingBuffer[' + ','.join(str(self[i]) for i in range(self.num_records)) + ']'

    def __repr__(self):
        return str(self)


if __name__ == '__main__':
    rec = NcclRecord(7, 20000)
    print(rec.data)
    d = {}
    # print(rec.data[...])
    j = 0
    for i in rec:
        if i[3] == 0:
            continue
        if i[4] not in d:
            d[i[4]] = [[i[3]], [i[2]]]
        else:
            d[i[4]][0].append(i[3])
            d[i[4]][1].append(d[i[4]][1][-1] + i[2])
    cs = ['red', 'g', 'b', 'black']
    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for gpu_id, (times, bytes) in d.items():
        times = np.array(times).astype(np.float64) / 1000000.0
        bytes = np.array(bytes) / (1024 * 1024)
        dt = times[1:] - times[:-1]
        ax1.plot(times, bytes, c=cs[gpu_id], label=f'GPU_{gpu_id}')
        ax2.scatter(np.arange(len(dt)), dt, s=5, c=cs[gpu_id], label=f'GPU_{gpu_id}', alpha=0.2)
        print(gpu_id, len(times))
    ax1.set_xlabel("Time / s")
    ax1.set_ylabel("Allreduced Size / MB")
    ax2.set_xlabel("# of Allreduce calls")
    ax2.set_ylabel(r"$\Delta$ t / s")
    plt.legend()
    plt.savefig("dt.png")
    # pid, (uint64_t)0, count, (uint64_t)(call_time * 1000), (uint64_t)(dev_id), buff1Addr, buff2Addr
    # plt.
