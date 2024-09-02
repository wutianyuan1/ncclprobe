import subprocess
import time
import torch
import torch.distributed as dist
from redis import StrictRedis
from dp_planner import PerformanceMetric, get_time_array, solve_dp


def set_gpu_frequency(node_id, gpu_id, duration, sim_factor=0.1, redis_cli=None, frequency=100, version=1):
    #### fail-slow by nvidia-smi: Previleged!!! ####
    # cmd = f"nvidia-smi -i {gpu_id} -lgc {frequency}"
    # try:
    #     subprocess.run(cmd, shell=True, check=True)
    #     print(f"Set GPU frequency to {frequency} MHz for GPU {gpu_id}")
    # except subprocess.CalledProcessError as e:
    #     print(f"Failed to set GPU frequency: {e}")

    #### fail-slow by computation process: Not good!!! ####
    # cmd = f"python comp_worker.py --device {gpu_id} --duration {duration * sim_factor}"
    # proc = subprocess.Popen(cmd, shell=True)

    ####  fail-slow by set model hook ####
    if isinstance(node_id, list):
        for nid in node_id:
            assert isinstance(nid, int)
            global_rank = nid * torch.cuda.device_count() + gpu_id
            delay_time = 0.075
            print(f"Set delay for global rank {global_rank}, delay_time = {delay_time}")
            redis_cli.set(f"delay_time_{global_rank}", delay_time)
    else:
        assert isinstance(node_id, int)
        global_rank = node_id * torch.cuda.device_count() + gpu_id
        delay_time = 0.075
        print(f"Set delay for global rank {global_rank}, delay_time = {delay_time}")
        redis_cli.set(f"delay_time_{global_rank}", delay_time)

    reaction_time = 30 + delay_time * 100  # seconds

    # wait duration
    time.sleep(reaction_time * sim_factor)

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

    compute_time = {
        i: PerformanceMetric(65, 65, 65, 0.01)
        for i in range(num_dps)
    }
    if isinstance(node_id, int):
        compute_time[my_dp_rank] = PerformanceMetric(515, 515, 515, 0.01)
    else:
        for nid in node_id:
            nid_dp_rank = nid % num_dps
            compute_time[nid_dp_rank] = PerformanceMetric(515, 515, 515, 0.01)

    time_array = get_time_array(redis_cli, compute_time)
    print("Iter times:", time_array)
    dp_ret = solve_dp(time_array, micro_bsz, global_bsz)
    print(f"DP: {dp_ret}, version={version}")
    redis_cli.set('batch_distribution', str(dp_ret))
    redis_cli.set("dp_version", version)

    time.sleep((duration - reaction_time) * sim_factor)

    #### End of fail-slow by nvidia-smi: Previleged!!! ####
    # cmd = f"nvidia-smi -i {gpu_id} -lgc 3000"
    # try:
    #     subprocess.run(cmd, shell=True, check=True)
    #     print(f"Set GPU frequency back to 3000 MHz for GPU {gpu_id}")
    # except subprocess.CalledProcessError as e:
    #     print(f"Failed to set GPU frequency: {e}")

    #### End of fail-slow by computation process: Not good!!! ####
    # proc.wait()

    #### End of fail-slow by set model hook ####
    if isinstance(node_id, list):
        for nid in node_id:
            assert isinstance(nid, int)
            global_rank = nid * torch.cuda.device_count() + gpu_id
            print(f"End of delay for global rank {global_rank}")
            redis_cli.set(f"delay_time_{global_rank}", 0)
    else:
        print(f"End of delay for global rank {global_rank}")
        redis_cli.set(f"delay_time_{global_rank}", 0)

    # Adjust to fair DP
    time.sleep(reaction_time * sim_factor / 2)
    fair_dp = [global_bsz // (micro_bsz * num_dps) for _ in range(num_dps)]
    print("Fair DP: ", fair_dp)
    redis_cli.set('batch_distribution', str(fair_dp))
    redis_cli.set("dp_version", version + 1)


def set_computation_slow(task_id, start, duration, node_id, gpu_id, st, ip_table, sim_factor, version):
    my_rank = dist.get_rank()
    if isinstance(node_id, int):
        if my_rank != node_id:
            return
    if isinstance(node_id, list):
        if my_rank != node_id[0]:
            return
    time.sleep(start * sim_factor)
    redis_cli = StrictRedis(host=ip_table[0])
    print(f"[t={time.time() - st}] Comp task {task_id}, start={start}, duration={duration}, GPU={gpu_id}")
    set_gpu_frequency(node_id, gpu_id, duration, sim_factor=sim_factor, redis_cli=redis_cli, frequency=100, version=version)
    print(f"[t={time.time() - st}] Comp task {task_id} done!")
