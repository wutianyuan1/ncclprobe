import time
import argparse
import subprocess
import redis
import logging
from typing import List, Dict
from .global_topo import GlobalTopo
from .task_split import split_ring, split_tree
from .global_analyzer import GlobalAnalyzer, PerformanceMetric, CommunicatorClique
from .communicator import Communicator, deserialize_communicator_from_redis


class ControlState:
    STATE_MONITOR  = 0
    STATE_PROFILE  = 1
    STATE_VALIDATE = 2


class ProcessRole:
    ROLE_SENDER = 0
    ROLE_RECVER = 1
    ROLE_ACKED  = 10086

class GlobalServer(object):
    def __init__(self, log_path, master_addr, port) -> None:
        log_ip = master_addr.replace('.', '_')
        logging.basicConfig(filename=log_path + f'/global_controller_{log_ip}.log')
        logging.getLogger().setLevel(logging.INFO)

        self.master_addr = master_addr
        self.port = port
        # start server & client
        self.redis_server_prog = self.start_redis_server()
        self.storage = redis.StrictRedis(host=master_addr, port=port, db=0)
        self.analyzer = GlobalAnalyzer(self.storage)
        self.precheck = False
        self.set_monitoring()

    def __del__(self):
        self.redis_server_prog.terminate()
        logging.critical("[GlobalController] Stop server!")
    
    def check_failslow_events(self):
        num_failslow_events = self.storage.llen("failslow_ranks")
        ret = self.storage.lrange("failslow_ranks", 0, num_failslow_events)
        return [str(i) for i in ret]
    
    def get_communicators(self):
        comms: List[Communicator] = []
        for key in self.storage.scan_iter("Communicator_*"):
            comm = deserialize_communicator_from_redis(
                self.storage.get(key)
            )
            comms.append(comm)
        return comms

    def pause_training(self):
        self.storage.set("control_state", str(ControlState.STATE_VALIDATE))

    def set_profiling(self):
        self.storage.set("control_state", str(ControlState.STATE_PROFILE))
    
    def set_monitoring(self):
        self.storage.set("control_state", str(ControlState.STATE_MONITOR))
    
    def dispatch_comm_task(self, task, comm_addrs, group2global):
        senders, recvers = task
        for (s, r) in zip(senders, recvers):
            sender_global_rank = group2global[s]
            recver_global_rank = group2global[r]
            self.storage.set(
                f"validtask_rank_{sender_global_rank}", f"{ProcessRole.ROLE_SENDER}_{r}_{comm_addrs[s]}"
            )
            self.storage.set(
                f"validtask_rank_{recver_global_rank}", f"{ProcessRole.ROLE_RECVER}_{s}_{comm_addrs[r]}"
            )
        # wait unitl all ranks acked the task
        for (s, r) in zip(senders, recvers):
            sender_global_rank = group2global[s]
            recver_global_rank = group2global[r]
            while True:
                sender_ret = self.storage.get(f"validtask_rank_{sender_global_rank}").decode()
                recver_ret = self.storage.get(f"validtask_rank_{recver_global_rank}").decode()
                if sender_ret == 'TASK_ACKED' and recver_ret == 'TASK_ACKED':
                    break
                time.sleep(1)
        logging.info(f"task {task} dispatched!")

    def collect_comm_task_results(self, task, group2global):
        senders, recvers = task
        all_ranks = set(senders + recvers)
        results = {}
        while len(results) != len(all_ranks):
            for group_rank in all_ranks:
                r = group2global[group_rank]
                res = self.storage.get(f"validtask_rank_{r}_result")
                if res is not None:
                    results[r] = PerformanceMetric.from_bytes(res)
                    logging.info(f"Result of rank {r} collected = {results[r]}!!")
                    self.storage.delete(f"validtask_rank_{r}_result")
            time.sleep(0.5)
        return results

    def start_redis_server(self):
        try:
            redis_prog = subprocess.Popen(["redis-server", "--bind", self.master_addr, "--save", "\"\"", "--appendonly", "no"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            logging.error("Cannot find redis on the server, exiting...")
            exit(1)
        time.sleep(1)
        return redis_prog
    
    def validate_ring(self, ring, comm_addrs, group2global):
        tasks = split_ring(ring)
        logging.info(f"validating ring:{ring}, the corresponding tasks are {tasks}")
        for task in tasks:
            self.dispatch_comm_task(task, comm_addrs, group2global)
            self.collect_comm_task_results(task, group2global)
            logging.info("*" * 20 + "task completed!" + "*" * 20)

    def validate_tree(self, tree, comm_addrs, group2global):
        tasks = split_tree(tree)
        logging.info(f"validating tree:{tree}, the corresponding tasks are {tasks}")
        for task in tasks:
            # skip empty tasks
            if len(task[0]) == 0:
                continue
            self.dispatch_comm_task(task, comm_addrs, group2global)
            self.collect_comm_task_results(task, group2global)
            logging.info("*" * 20 + "task completed!" + "*" * 20)

    def validate_computation(self):
        world_size = self.analyzer.world_size
        for i in range(world_size):
            self.storage.set(
                f"validtask_rank_{i}", "ComputationTest"
            )

        # wait unitl all ranks acked the task
        for i in range(world_size):
            while True:
                comp_ret = self.storage.get(f"validtask_rank_{i}").decode()
                if comp_ret == 'TASK_ACKED':
                    break
                time.sleep(1)
        logging.info(f"Computation tasks are dispatched!")

        results = {}
        while len(results) != world_size:
            for r in range(world_size):
                res = self.storage.get(f"validtask_rank_{r}_result")
                if res is not None:
                    results[r] = PerformanceMetric.from_bytes(res)
                    logging.info(f"Computation result of rank {r} collected = {results[r]}!!")
                    self.storage.delete(f"validtask_rank_{r}_result")
            time.sleep(0.5)

    def handle_failslow(self, failslow_events):
        # Fisrt, we enables profile mode to enable CUDA events and collect kernel durations
        self.set_profiling()
        # (1) wait and get profile results
        perfs = self.analyzer.wait_and_build_performance_map()
        comms = self.get_communicators()
        cliques = self.analyzer.build_comm_cliques(comms)
        # (2) analyze profile results and determine which ring/tree to validate
        slow_cliques = self.analyzer.find_slow_clique(perfs, cliques)
        validate_topos = [(c, GlobalTopo(c.comms)) for c in slow_cliques]
        # (3) dispatch validation jobs and collect validation results
        self.pause_training()
        time.sleep(1)

        # (3.1) check computation
        self.validate_computation()
        # (3.2) check communication
        for clique, topo in validate_topos:
            # Skip single-element ring/trees
            if len(topo.rings[0]) == 1 or len(topo.trees[0]) == 1:
                continue
            comm_addrs = {}
            group2global = {}
            for cm in clique.comms:
                comm_addrs[cm.group_rank] = cm.comm_addr
                group2global[cm.group_rank] = cm.global_rank
            self.validate_ring(topo.rings[0], comm_addrs, group2global)
            self.validate_tree(topo.trees[0], comm_addrs, group2global)
        # Finally, clear the failed slow events and resume all jobs to monitoring state
        self.storage.ltrim("failslow_ranks", 1, 0)
        self.set_monitoring()
    
    def do_precheck(self):
        # In pre-check, we validate all rings/trees that NCCL built
        comms = self.get_communicators()
        all_cliques: Dict[int, CommunicatorClique] =\
            self.analyzer.build_comm_cliques(comms)
        unique_cliques: List[CommunicatorClique] = []
        unique_ids = set()
        for c in all_cliques.values():
            if c.clique_id not in unique_ids and len(c.ranks) > 1:
                unique_ids.add(c.clique_id)
                unique_cliques.append(c)
        logging.info(f"All rings: {unique_cliques}")
        validate_topos = [(c, GlobalTopo(c.comms)) for c in unique_cliques]
        # Dispatch validation jobs and collect validation results
        self.pause_training()
        time.sleep(1)

        # First, check computation performance
        self.validate_computation()

        for clique, topo in validate_topos:
            # Skip single-element ring/trees
            if len(topo.rings[0]) == 1 or len(topo.trees[0]) == 1:
                continue
            comm_addrs = {}
            group2global = {}
            for cm in clique.comms:
                comm_addrs[cm.group_rank] = cm.comm_addr
                group2global[cm.group_rank] = cm.global_rank
            self.validate_ring(topo.rings[0], comm_addrs, group2global)
            self.validate_tree(topo.trees[0], comm_addrs, group2global)
        # Finally, clear the failed slow events and resume all jobs to monitoring state
        self.set_monitoring()

    def run(self):
        try:
            self.storage.set("global_controller", "OK")
            while True:
                time.sleep(5)
                if not self.precheck:
                    # wait additional 5 seconds to ensure all comms are built.
                    time.sleep(5)
                    self.precheck = True
                    self.do_precheck()
                    self.storage.set("precheck_done", '1')
                failslow_events = self.check_failslow_events()
                if len(failslow_events) != 0:
                    logging.info(f"[GlobalController] failslow events are reported from local: {failslow_events}")
                    self.handle_failslow(failslow_events)
        except KeyboardInterrupt:
            logging.critical("[GlobalController] Stop server running!")
            return


def main():
    parser = argparse.ArgumentParser("Global side controller of fail-slow detection")
    parser.add_argument("-m", "--master_addr", default="127.0.0.1", type=str)
    parser.add_argument("-p", "--port", default=6379, type=int)
    parser.add_argument("-o", "--output_path", default="/workspace/ncclprobe/logs/", type=str)
    args = parser.parse_args()
    server = GlobalServer(args.output_path, args.master_addr, args.port)
    server.run()


if __name__ == "__main__":
    main()
