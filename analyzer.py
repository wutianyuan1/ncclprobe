import re
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from slow_detection import find_period, find_performance_drop
from multiprocessing import shared_memory, resource_tracker

CONFIG = {}
OPS = ['Send', 'Recv', 'Bcast', 'Broadcast', 'AllGather', 'ReduceScatter', 'AllReduce']
SIZEOF_INT64 = 8


def sizestr(size):
    if size < 1024:
        return str(size)
    elif size < 1024 * 1024:
        return str(size // 1024) + "KB"
    elif size < 1024 * 1024 * 1024:
        return str(size // (1024**2)) + "MB"
    else:
        return str(size // (1024**3)) + "GB"


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
             'aux', 'num_devices']

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
    record_df = pd.DataFrame([i for i in record], columns=field_keys)
    f, axs = plt.subplots(1, 4, sharey='row', figsize=(12, 3))
    colors = ['powderblue', 'grey', 'lightblue', 'red', 'lightyellow', 'pink', 'lightgreen']
    stats = []
    for gpu_id, per_gpu_calls in record_df.groupby('device'):
        dts = {}
        for op_id, per_op_calls in per_gpu_calls.groupby("call_number"):
            per_op_calls = per_op_calls.sort_values(by=['event_id'])
            ts = per_op_calls['call_time'].to_numpy()
            dt = (ts[1:] - ts[:-1]) / (1000 * 1000)  # convert microsecond to second
            # skip the rare calls
            if len(dt) < 50:
                continue
            print(gpu_id, OPS[op_id], per_op_calls['count'])
            dts[op_id] = dt
            stats.append([gpu_id, OPS[op_id], len(dt), np.mean(dt), np.std(dt)])
        bplot = axs[gpu_id].boxplot(
            list(dts.values()), notch=True, vert=True, patch_artist=True,
            showfliers=False, labels=[OPS[key] for key in dts])
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        axs[gpu_id].set_xlabel("NCCL Operation")
        axs[gpu_id].set_title(f"GPU {gpu_id}")
        if gpu_id == 0:
            axs[gpu_id].set_ylabel(r"$\Delta$ t / s")
    stats_df = pd.DataFrame(stats, columns=['GPU_ID', 'OP_ID', 'LEN', 'INTERVAL_MEAN', 'INTERVAL_STD'])
    print(stats_df)
    stats_df.to_csv("logs/interval_stats.csv")
    plt.tight_layout()
    plt.savefig("figs/new_dt.png")


def find_slow_events(record: NcclRecord):
    field_keys = record.attrs + [f"device_{i}" for i in range(CONFIG['MAX_DEVS'] - 1)] + ['event_id']
    record_df = pd.DataFrame([i for i in record], columns=field_keys)
    f, axs = plt.subplots(1, 4, sharey='row', figsize=(12, 3))
    colors = ['powderblue', 'grey', 'pink', 'green']

    for (gpu_id, per_gpu_record) in record_df.groupby("device"):
        per_gpu_record.sort_values(by='event_id', inplace=True)
        call_time = per_gpu_record['call_time'].to_numpy()
        call_id = per_gpu_record['call_number'].to_numpy()
        start, period = find_period(call_id, nlags=30, significance_level=0.8)
        print(gpu_id, start, period)
        pargs = {"ax": axs[gpu_id], "color": colors[gpu_id], "label": f"GPU_{gpu_id}",
                 "xlabel": "Execution Time / us", "ylabel": "Iteration Time / us"}
        find_performance_drop(call_id, call_time, period, start, plot=True, plot_args=pargs)
    plt.tight_layout()
    plt.savefig("figs/period.png")


if __name__ == '__main__':
    logging.basicConfig(filename='logs/analyzer.log')
    logging.getLogger().setLevel(logging.INFO)
    load_config()
    record = NcclRecord(CONFIG['NUM_FIELDS'], CONFIG['BUFFER_SIZE'])
    find_slow_events(record)
    # plot_call_interval(record)
