import time
import os
import subprocess
import torch.distributed as dist
import multiprocessing as mp
import numpy as np
from redis import StrictRedis


def set_communication_slow(task_id, start, duration, src_node, src_gpu, dst_node, dst_gpu, master_port, st, ip_table, sim_factor):
    my_rank = dist.get_rank()
    if my_rank != src_node and my_rank != dst_node:
        return
    time.sleep(start * sim_factor)
    master = ip_table[src_node]
    print(f"[t={time.time() - st}] Comm task {task_id}, rank={my_rank}, start={start}, duration={duration}, master={master} src [node={src_node}, GPU={src_gpu}], dst [node={dst_node}, GPU={dst_gpu}]")
    cmd_base = "MASTER_ADDR={} MASTER_PORT={} WORLD_SIZE=2 RANK={} python comm_worker.py --tensor-size 100 --duration {}"
    if my_rank == src_node:
        os.system(f"python adjust_pp.py {ip_table[0]} {dist.get_rank()} {dist.get_world_size()} {40} 0 &")
        cmd = cmd_base.format(master, master_port, 0, duration)
        subprocess.run(cmd, shell=True)
        os.system(f"python adjust_pp.py {ip_table[0]} {dist.get_rank()} {dist.get_world_size()} {0} 1 &")
    elif my_rank == dst_node:
        cmd = cmd_base.format(master, master_port, 1, duration)
        subprocess.run(cmd, shell=True)
    print(f"[t={time.time() - st}] Comm task {task_id} done!!")
