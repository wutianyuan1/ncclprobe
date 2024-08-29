import os
import time
import numpy as np
import pickle
import random
import redis
import socket
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt
from slow_comp import set_computation_slow
from slow_comm import set_communication_slow


sim_factor = 1


def print_error(e):
    print("Error:", e)


def generate_slow_durations(nnodes=1, load_fn=None):
    if load_fn is None:
        ret = []
        for node_id in range(nnodes):
            timestamps = []
            current_time = 0
            while current_time < 3600:
                # poisson process
                interarrival_time = np.random.exponential(scale=300)
                start_time = current_time + interarrival_time
                duration = max(1, int(np.random.normal(loc=120, scale=20)))
                reason = np.random.randint(0, 2)  # 0: comp; 1: comm
                if reason == 0:
                    gpu_1 = np.random.randint(0, 8)
                    gpu_2 = gpu_1
                    node_1 = node_id
                    node_2 = node_id
                else:
                    gpu_1 = np.random.randint(0, 8)
                    gpu_2 = np.random.randint(0, 8)
                    node_1 = node_id
                    node_2 = np.random.choice([i for i in range(nnodes)])
                timestamps.append((start_time, duration, node_1, gpu_1, node_2, gpu_2, reason))
                current_time = start_time + duration + 60
            ret.append(np.array(timestamps, dtype=int))
        with open(f"{nnodes}node.pkl", 'wb') as f:
            pickle.dump(ret, f)
        return ret
    else:
        with open(load_fn, 'rb') as f:
            return pickle.load(f)


def plot_slow_node(data):
    coords = [[(0, 3000)] for _ in range(8)]
    max_t = 0
    for line in data:
        start, dur, gpu = line[0], line[1], line[-4]
        coords[gpu].append((start, 3000))
        coords[gpu].append((start + 0.01, 100))
        coords[gpu].append((start + dur, 100))
        coords[gpu].append((start + dur + 0.01, 3000))
        max_t = start + dur + 100
    for i in range(len(coords)):
        coords[i].append((max_t, 3000))
    for i, line in enumerate(coords):
        line = np.array(line)
        plt.plot(line[:, 0], line[:, 1], label=f"GPU{i}")
    plt.xlabel("Time / s")
    plt.ylabel("Frequency / MHz")
    plt.legend()
    plt.savefig("1node.png")



def main():
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])
    rank = int(os.environ['RANK'])

    # Initialize the process group
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )

    # node_id is rank_id because each node only run one proc
    node_id = dist.get_rank()
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    ip_tensor = torch.tensor([int(i) for i in ip.split(".")], dtype=torch.int32)
    print(f"My rank: {node_id}, IP={ip}, ip_tensor={ip_tensor}")

    ip_table = [torch.zeros(4, dtype=torch.int32) for _ in range(world_size)]
    dist.all_gather(ip_table, ip_tensor)
    ip_table = [".".join([str(i) for i in line.tolist()]) for line in ip_table]
    print(f"IP table: {ip_table}")

    # reset GPU frequency
    os.system("nvidia-smi -lgc 3000")
    time.sleep(1)


    # Get the traces to run
    # all_data = generate_slow_durations(load_fn=f'{1}node.pkl')
    # my_data = all_data[0]
    my_data = [
        # [1, 5, 0, 0, 2, 0, 1],
        # [9, 5, 1, 0, 3, 0, 1],
        [25, 40, 2, 0, 3, 0, 1],
        [90, 30, 2, 0, 2, 0, 0]
    ]
    print(my_data)


    pool = mp.Pool(len(my_data))
    rets = []
    version = 1
    st = time.time()
    for (i, line) in enumerate(my_data):
        start, duration, src_node, src_gpu, dst_node, dst_gpu, reason = line
        if reason == 0:
            ret = pool.apply_async(set_computation_slow,
                                   args=(i, start, duration, src_node, src_gpu, st, ip_table, sim_factor, version),
                                   error_callback=print_error)
            version += 2
        else:
            ret = pool.apply_async(set_communication_slow,
                                   args=(i, start, duration, src_node, src_gpu, dst_node, dst_gpu, 9969+i, st, ip_table, sim_factor),
                                   error_callback=print_error)
        rets.append(ret)
    pool.close()
    pool.join()

    for r in rets:
        print("wait", r)
        r.wait()

    x = dist.all_reduce(torch.tensor([1]))
    print("allreduce2 done", x)

    # Finalize the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
