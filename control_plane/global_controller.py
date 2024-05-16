import time
import argparse
import subprocess
import redis
import struct
import logging
from typing import List
from global_topo import GlobalTopo
from task_split import split_ring
from global_analyzer import GlobalAnalyzer
from communicator import Communicator, deserialize_communicator_from_redis


class ControlState:
    STATE_MONITOR  = 0
    STATE_PROFILE  = 1
    STATE_VALIDATE = 2


class ProcessRole:
    ROLE_SENDER = 0
    ROLE_RECVER = 1
    ROLE_ACKED  = 10086


class PerformanceMetric(object):
    def __init__(self, min_lat, max_lat, avg_lat) -> None:
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.avg_lat = avg_lat

    @classmethod
    def from_bytes(cls, redis_data):
        data = struct.unpack('ddd', redis_data[:24])
        return PerformanceMetric(*data)
    
    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"(Perf: min={self.min_lat}, max={self.max_lat}, avg={self.avg_lat})"


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
        self.resume_jobs()

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
            if comm.num_devices != 1:
                comms.append(comm)
        return comms

    def pause_jobs(self):
        self.storage.set("control_state", str(ControlState.STATE_VALIDATE))

    def start_profile(self):
        self.storage.set("control_state", str(ControlState.STATE_PROFILE))
    
    def resume_jobs(self):
        self.storage.set("control_state", str(ControlState.STATE_MONITOR))
    
    def dispatch_task(self, task):
        senders, recvers = task
        for (s, r) in zip(senders, recvers):
            self.storage.set(
                f"validtask_rank_{s}", f"{ProcessRole.ROLE_SENDER}_{r}"
            )
            self.storage.set(
                f"validtask_rank_{r}", f"{ProcessRole.ROLE_RECVER}_{s}"
            )
        logging.info(f"task {task} dispatched!")

    def get_task_result(self, task):
        senders, recvers = task
        all_ranks = set(senders + recvers)
        results = {}
        while len(results) != len(all_ranks):
            for r in all_ranks:
                res = self.storage.get(f"validtask_rank_{r}_result")
                if res is not None:
                    results[r] = PerformanceMetric.from_bytes(res)
                    logging.info(f"Result of rank {r} collected = {results[r]}!!")
            time.sleep(0.5)
        return results

    def start_redis_server(self):
        try:
            redis_prog = subprocess.Popen(["redis-server", "--save", "\"\"", "--appendonly", "no"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            logging.error("Cannot find redis on the server, exiting...")
            exit(1)
        time.sleep(1)
        return redis_prog
    
    def validate_ring(self, ring):
        tasks = split_ring(ring)
        print(ring, tasks)
        for task in tasks:
            self.dispatch_task(task)
            self.get_task_result(task)
            logging.info("*" * 20 + "task completed!" + "*" * 20)

    def handle_failslow(self, failslow_events):
        # Fisrt, we enables profile mode to enable CUDA events and collect kernel durations
        self.start_profile()
        # TODO: (1) get profile results
        # TODO: (2) analyze profile results and determine which ring/tree to validate
        # TODO: (3) dispatch validation jobs and collect validation results
        time.sleep(20)
        # Finally, clear the failed slow events and resume all jobs to monitoring state
        self.storage.ltrim("failslow_ranks", 1, 0)
        self.resume_jobs()

    def run(self):
        try:
            self.storage.set("global_controller", "OK")
            while True:
                time.sleep(5)
                comms = self.get_communicators()
                cliques = self.analyzer.build_comm_cliques(comms)
                for k, v in cliques.items():
                    print("Comm", k, [i.global_rank for i in v])
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
