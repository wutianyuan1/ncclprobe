import subprocess
import time
import torch
import torch.distributed as dist
from redis import StrictRedis
from dp_planner import PerformanceMetric, get_time_array, solve_dp


def set_gpu_frequency(gpu_id, duration, sim_factor=0.1, redis_cli=None, frequency=100, version=1):
    # fail-slow
    cmd = f"nvidia-smi -i {gpu_id} -lgc {frequency}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Set GPU frequency to {frequency} MHz for GPU {gpu_id}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set GPU frequency: {e}")
    
    dp_data = redis_cli.get("0_dp")
    if dp_data is not None:
        dp_data = dp_data.decode().split("_")
        num_dps = len(dp_data)
    else:
        num_dps = dist.get_world_size()
    my_dp_rank = dist.get_rank() % num_dps
    micro_bsz = redis_cli.get("micro_batch_size")
    global_bsz = redis_cli.get("global_batch_size")
    if micro_bsz is not None and global_bsz is not None:
        micro_bsz = int(micro_bsz.decode())
        global_bsz = int(global_bsz.decode())
    else:
        micro_bsz, global_bsz = 2, 256
    print(f"DP world size: {num_dps}, my_dp_rank: {my_dp_rank}, micro_bsz: {micro_bsz}, global_bsz: {global_bsz}")

    # wait duration
    time.sleep(5 * sim_factor)
    compute_time = {
        i: PerformanceMetric(65, 65, 65, 0.01)
        for i in range(num_dps)
    }
    compute_time[my_dp_rank] = PerformanceMetric(515, 515, 515, 0.01)

    time_array = get_time_array(redis_cli, compute_time)
    print("Iter times:", time_array)
    dp_ret = solve_dp(time_array, micro_bsz, global_bsz)
    print("DP:", dp_ret)
    time.sleep((duration - 5) * sim_factor)
    redis_cli.set('batch_distribution', str(dp_ret))
    redis_cli.set("dp_version", version)

    # back to normal
    cmd = f"nvidia-smi -i {gpu_id} -lgc 3000"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Set GPU frequency back to 3000 MHz for GPU {gpu_id}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set GPU frequency: {e}")

    time.sleep(5 * sim_factor)
    fair_dp = [global_bsz // (micro_bsz * num_dps) for _ in range(num_dps)]
    print("Fair DP: ", fair_dp)
    redis_cli.set('batch_distribution', str(fair_dp))
    redis_cli.set("dp_version", version + 1)


def set_computation_slow(task_id, start, duration, node_id, gpu_id, st, ip_table, sim_factor, version):
    my_rank = dist.get_rank()
    if my_rank != node_id:
        return
    time.sleep(start * sim_factor)
    redis_cli = StrictRedis(host=ip_table[0])
    print(f"[t={time.time() - st}] Comp task {task_id}, start={start}, duration={duration}, GPU={gpu_id}")
    set_gpu_frequency(gpu_id, duration, sim_factor=sim_factor, redis_cli=redis_cli, frequency=100, version=version)
    print(f"[t={time.time() - st}] Comp task {task_id} done!")
