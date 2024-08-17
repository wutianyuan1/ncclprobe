import redis
import numpy as np
import cvxpy as cp
from typing import Dict
from .global_analyzer import PerformanceMetric


def get_time_array(redis_client: redis.StrictRedis, compute_time: Dict[int, PerformanceMetric], threshold: float = 1.1):
    # normal iteration time
    min_iter_time = int(redis_client.get('min_iter_time').decode())
    # iteration time after fail-slow
    slow_iter_time = int(redis_client.get("cur_iter_time").decode())
    # find the stats of compute validation
    time_array = np.zeros(len(compute_time), dtype=np.float32)
    vals = [i.avg_lat for i in compute_time.values()]
    min_compute_time, max_compute_time = np.min(vals), np.max(vals)
    median_compute_time = np.median(vals)
    for rank in compute_time:
        rank_lat = compute_time[rank].avg_lat
        # faster than 1.1*median_compute => not fail slow
        if rank_lat <= threshold * median_compute_time:
            time_array[rank] = min_iter_time
        # fail slow => compute its slowdown percentage
        else:
            # assume compute time is linear to iteration time: iter=k*compute+b
            slope = (slow_iter_time - min_iter_time) / (max_compute_time - min_compute_time)
            intercept = slow_iter_time - slope * max_compute_time
            time_array[rank] = slope * rank_lat + intercept
    return time_array


def solve(time_array: np.ndarray, micro_bsz: int, global_bsz: int):
    num_dp_groups = len(time_array)
    num_microbatches = cp.Variable(shape=num_dp_groups, integer=True)
    N_t = cp.multiply(num_microbatches, time_array)
    avg_N_t = cp.mean(N_t)
    variance = cp.sum_squares(N_t - avg_N_t)
    constraints = [
        num_microbatches >= 1,  # Each N_i must be positive
        cp.sum(num_microbatches * micro_bsz) == global_bsz  # The batch size sum constraint
    ]
    problem = cp.Problem(cp.Minimize(variance), constraints)
    problem.solve(solver=cp.ECOS_BB)
    num_microbatches = [round(i) for i in num_microbatches.value]
    return num_microbatches


if __name__ == '__main__':
    redis_host = 'localhost'
    redis_port = 6379
    client = redis.StrictRedis(redis_host, redis_port, db=0)
    compute_time = {
        0: PerformanceMetric(65, 65, 65, 0.01),
        1: PerformanceMetric(65, 65, 65, 0.01),
        2: PerformanceMetric(65, 80, 75, 5.01),
        3: PerformanceMetric(65, 65, 65, 0.01),
        4: PerformanceMetric(65, 65, 65, 0.01),
        5: PerformanceMetric(65, 200, 150, 50.01),
        6: PerformanceMetric(65, 65, 65, 0.01),
        7: PerformanceMetric(65, 65, 65, 0.01)
    }
    time_array = get_time_array(client, compute_time)
    print(time_array)
    ret = solve(time_array, 2, 256)
    print(ret)
