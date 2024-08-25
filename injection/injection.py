import os
import subprocess
import time
import numpy as np
import pickle
import random
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt


sim_factor = 0.002


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


def set_gpu_frequency(gpu_id, duration, frequency=100):
    # fail-slow
    cmd = f"nvidia-smi -i {gpu_id} -lgc {frequency}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Set GPU frequency to {frequency} MHz for GPU {gpu_id}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set GPU frequency: {e}")
    
    # wait duration
    time.sleep(duration * sim_factor)

    # back to normal
    cmd = f"nvidia-smi -i {gpu_id} -lgc 3000"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Set GPU frequency back to 3000 MHz for GPU {gpu_id}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set GPU frequency: {e}")


def set_computation_slow(task_id, start, duration, gpu_id):
    print(f"Lock task {task_id}, start={start}, duration={duration}, GPU={gpu_id}")
    time.sleep(start * sim_factor)
    set_gpu_frequency(gpu_id, duration)


def set_communication_slow(task_id, start, duration, src_node, src_gpu, dst_node, dst_gpu):
    my_rank = dist.get_rank()
    if my_rank != src_node and my_rank != dst_node:
        return
    print(f"Comm task {task_id}, rank={my_rank}, start={start}, duration={duration}, src [node={src_node}, GPU={src_gpu}], dst [node={dst_node}, GPU={dst_gpu}]")
    time.sleep(start * sim_factor)
    # TODO: perform communication with dst
    if my_rank == src_node:
        print("src!!")
        data = torch.zeros((1024, 1024, 20), dtype=torch.float32, device=f'cuda:{src_gpu}')
        print("111")
        dist.send(data, dst=dst_node)
        print("send!!")
    elif my_rank == dst_node:
        print("dst!!")
        data = torch.zeros((1024, 1024, 20), dtype=torch.float32, device=f'cuda:{dst_gpu}')
        ret = dist.recv(data, src=src_node)
        print("recv!!")


def main():
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])
    rank = int(os.environ['RANK'])

    # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )

    # node_id is rank_id because each node only run one proc
    node_id = dist.get_rank()
    print(f"My rank: {node_id}")

    # reset GPU frequency
    os.system("nvidia-smi -lgc 3000")
    time.sleep(1)

    # Get the traces to run
    all_data = generate_slow_durations(load_fn=f'{1}node.pkl')
    my_data = all_data[0]
    print(my_data)

    pool = mp.Pool(len(my_data))

    for (i, line) in enumerate(my_data):
        start, duration, src_node, src_gpu, dst_node, dst_gpu, reason = line
        if reason == 0:
            pool.apply_async(set_computation_slow, args=(i, start, duration, src_gpu))
        else:
            pool.apply_async(set_communication_slow, args=(i, start, duration, src_node, src_gpu, dst_node, dst_gpu), error_callback=print_error)
    pool.close()
    pool.join()

    # Finalize the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
