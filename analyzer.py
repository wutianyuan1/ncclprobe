import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import shared_memory, resource_tracker

CONFIG = {}
OPS = ['Send', 'Recv', 'Bcast', 'Broadcast', 'AllGather', 'ReduceScatter', 'AllReduce']
SIZEOF_INT64 = 8

def sizestr(size):
    if size < 1024:
        return str(size)
    elif size < 1024*1024:
        return str(size//1024) + "KB"
    elif size < 1024*1024*1024:
        return str(size//(1024**2)) + "MB"
    else:
        return str(size//(1024**3)) + "GB"


def load_config():
    global CONFIG
    with open("config.hpp") as header_file:
        for line in header_file.readlines():
            if line.startswith("#define"):
                line.rstrip()
                m = re.search(r'#define\s+([A-Za-z]\w+)\s+(.*)', line)
                if m:
                    content = m.group(2)
                    CONFIG[m.group(1)] = int(content, base=16 if '0x' in content else 10)
    with open("shm_storage.hpp") as storage_file:
        all_lines = storage_file.read().replace("\n", "")
        all_lines = re.sub(r" +", r" ", all_lines)
        m = re.search(r'Record\((uint64\_t\s[A-Za-z0-9_]+,\s*)*', all_lines)
        numfields = m.group(0).count(',')
        CONFIG['NUM_FIELDS'] = numfields + CONFIG['MAX_DEVS']
    print(CONFIG)


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
    attrs = ['call_number', 'count', 'buff1', 'buff2',
        'datatype', 'pid', 'call_time', 'device', 'caller',
        'aux', 'num_devices'
    ]
    def __init__(self, num_fields, max_records):
        remove_shm_from_resource_tracker()
        self.shm_size = (num_fields * max_records + CONFIG['METADATA_FIELDS']) * SIZEOF_INT64
        self.shm = shared_memory.SharedMemory(
            "ncclRecord", create=False, size=self.shm_size)
        self.data = np.frombuffer(self.shm.buf, np.int64)
        self.buffer = self.data[CONFIG['METADATA_FIELDS']:]
        self.num_fields = self.data[0]
        self.max_records = self.data[1]

    @property
    def num_records(self):
        return self.data[2]

    def get_profile_data(self, metric_name):
        if metric_name == 'event_id':
            return [i[-1] for i in self]
        metric_id = self.attrs.index(metric_name)
        ret = []
        for record in self:
            ret.append(record[metric_id])
        return np.array(ret)
    
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


def plot_call_interval(record: NcclRecord):
    field_keys = record.attrs + [f"device_{i}" for i in range(CONFIG['MAX_DEVS'] - 1)] + ['event_id']
    df = pd.DataFrame([i for i in record], columns=field_keys)
    print(df['num_devices'])
    f, axs = plt.subplots(1, 4, sharey='row', figsize=(12, 3))
    colors = ['black', 'grey', 'blue', 'red', 'yellow', 'pink', 'green']
    for gpu_id, per_gpu_calls in df.groupby('device'):
        for op_id, per_op_calls in per_gpu_calls.groupby("call_number"):
            # print(gpu_id, op_id, '='*60)
            # print(per_op_calls)
            xs, ys = [], []
            prev = {}
            per_op_calls = per_op_calls.sort_values(by=['event_id'])
            for _, row in per_op_calls.iterrows():
                cnt = sizestr(row['count'])
                t = row['call_time']
                xs.append(cnt)
                if cnt not in prev:
                    ys.append(0)
                else:
                    ys.append((t - prev[cnt])/(1000 * 1000))
                prev[cnt] = t
            axs[gpu_id].scatter(xs, ys, s=3, c=colors[op_id], label=OPS[op_id])
        axs[gpu_id].set_xlabel("Operation Size")
        if gpu_id == 0:
            axs[gpu_id].set_ylabel(r"$\Delta$ t / s")
        axs[gpu_id].legend()
        # axs[gpu_id].set_ylim(0, 3)
    plt.tight_layout()
    plt.savefig("new_dt.png")



if __name__ == '__main__':
    load_config()
    record = NcclRecord(CONFIG['NUM_FIELDS'], CONFIG['BUFFER_SIZE'])
    plot_call_interval(record)
