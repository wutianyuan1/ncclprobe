import logging
import time
from typing import Dict, List
from copy import deepcopy
from .global_analyzer import PerformanceMetric, CommunicatorClique
from .dp_planner import get_time_array, solve_dp
from .cost_estimator import CostEstimator


class MitigationPlan(object):
    def __init__(self, redis_client):
        self.client = redis_client
        self.dp_version = 0
        self.estimator = CostEstimator()
        self.rs_cost = self.estimator.get_restart_adjustment_cost()

    def group_comp_time_by_dp(self,
                              all_cliques: Dict[int, CommunicatorClique],
                              comp_results: Dict[int, PerformanceMetric]):
        comp_results = deepcopy(comp_results)
        # 1st pass: Synchronize all results within the same PP group
        for c in all_cliques.values():
            if c.is_pp:
                ts = [comp_results[i].avg_lat for i in c.ranks]
                max_ts = max(ts)
                for r in c.ranks:
                    comp_results[r] = PerformanceMetric(
                        max_ts, max_ts, max_ts, 0
                    )
        # 2nd pass: Synchronize all results within the same TP group
        for c in all_cliques.values():
            if c.is_tp:
                ts = [comp_results[i].avg_lat for i in c.ranks]
                max_ts = max(ts)
                for r in c.ranks:
                    comp_results[r] = PerformanceMetric(
                        max_ts, max_ts, max_ts, 0
                    )
        # 3rd pass: aggregate to DP group
        dp_results = {}
        for c in all_cliques.values():
            if c.is_dp:
                for (dpr, r) in enumerate(c.ranks):
                    dp_results[dpr] = comp_results[r]
        logging.info(f"DP computation results: {dp_results}")
        return dp_results

    def adjust_batchsize_distribution(self,
                                      comp_results: Dict[int, PerformanceMetric],
                                      cliques: Dict[int, CommunicatorClique]):
        comp_results_by_dp = self.group_comp_time_by_dp(cliques, comp_results)
        time_array = get_time_array(self.client, comp_results_by_dp)
        # Mitigation by adjusting batch size distribution across DP groups
        micro_bsz = self.client.get("micro_batch_size")
        global_bsz = self.client.get("global_batch_size")
        if micro_bsz is not None and global_bsz is not None:
            micro_bsz = int(micro_bsz.decode())
            global_bsz = int(global_bsz.decode())
            new_dp = solve_dp(time_array, micro_bsz, global_bsz)
            # microbatch changed
            self.dp_version += 1
            self.client.set('batch_distribution', str(new_dp))
            self.client.set("dp_version", self.dp_version)
        else:
            logging.warning(f"batch size not found: microbsz={micro_bsz}, globalbsz={global_bsz}")

    def adjust_pipeline_parallel(self,
                                 comm_cliques: Dict[int, CommunicatorClique],
                                 comm_tasks: List,
                                 comm_results: List[Dict[int, PerformanceMetric]]):
        pass

    def mitigate_failslow(self,
                          dp_check_interval: int,
                          comp_results: Dict[int, PerformanceMetric],
                          comm_cliques: Dict[int, CommunicatorClique],
                          comm_tasks: List,
                          comm_results: List[Dict[int, PerformanceMetric]]):
        slow_start = time.time()
        dp_adjusted, pp_adjusted = False, False
        slow_iter_time_init = float(self.client.get("cur_iter_time").decode())
        min_iter_time_init = float(self.client.get('min_iter_time').decode())
        if slow_iter_time_init >= min_iter_time_init * 1.1:
            fast_to_slow = True
        else:
            fast_to_slow = False
        while True:
            # normal iteration time
            min_iter_time = float(self.client.get('min_iter_time').decode())
            # iteration time after fail-slow
            slow_iter_time = float(self.client.get("cur_iter_time").decode())
            dp_cost = self.estimator.get_dp_adjustment_cost(dp_check_interval, min_iter_time, slow_iter_time)
            pp_cost = self.estimator.get_pp_adjustment_cost()
            time_since_slow = 1000 * (time.time() - slow_start)
            logging.info(f"[Mitigation Plan] DPcost={dp_cost}, PPcost={pp_cost}, time_since_slow={time_since_slow}, min_iter={min_iter_time}, slow_iter={slow_iter_time}")
            if time_since_slow >= dp_cost and not dp_adjusted:
                logging.info("[Mitigation Plan] Adjust DP")
                self.adjust_batchsize_distribution(comp_results, comm_cliques)
                dp_adjusted = True
            if time_since_slow >= pp_cost and not pp_adjusted:
                logging.info("[Mitigation Plan] Adjust PP")
                self.adjust_pipeline_parallel(comm_cliques, comm_tasks, comm_results)
                pp_adjusted = True
            # if this fail-slow ends, and performance backs to normal, break
            # if this performance change is from slow stage to normal stage, this will be skipped
            if slow_iter_time <= 1.1 * min_iter_time and fast_to_slow:
                break
            if time_since_slow >= 2 * pp_cost:
                break
            time.sleep(1)
