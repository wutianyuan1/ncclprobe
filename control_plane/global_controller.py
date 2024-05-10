import time
import argparse
import subprocess
import redis
from typing import List
from global_topo import GlobalTopo
from task_split import split_ring
from communicator import Communicator, deserialize_communicator_from_redis


class ControlState:
    STATE_MONITOR  = 0
    STATE_PROFILE  = 1
    STATA_VALIDATE = 2


class ProcessRole:
    ROLE_SENDER = 0
    ROLE_RECVER = 1


class GlobalServer(object):
    def __init__(self, master_addr, port) -> None:
        self.master_addr = master_addr
        self.port = port
        # start server & client
        self.redis_server_prog = self.start_redis_server()
        self.storage = redis.StrictRedis(host=master_addr, port=port, db=0)

    def __del__(self):
        self.redis_server_prog.terminate()
        print("Stop server!")
    
    def get_communicators(self):
        print("="*20)
        comms: List[Communicator] = []
        for key in self.storage.scan_iter("Communicator_*"):
            comm = deserialize_communicator_from_redis(
                self.storage.get(key)
            )
            if comm.num_devices != 1:
                comms.append(comm)
                print(comm)
        return comms

    def pause_jobs(self):
        self.storage.set("control_state", str(ControlState.STATA_VALIDATE))

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
        print(f"task {task} dispatched!")
        time.sleep(5)

    def start_redis_server(self):
        try:
            redis_prog = subprocess.Popen(["redis-server", "--save", "\"\"", "--appendonly", "no"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            print("Cannot find redis on the server, exiting...")
            exit(1)
        time.sleep(1)
        return redis_prog
    
    def validate_ring(self, ring):
        tasks = split_ring(ring)
        print(ring, tasks)
        for task in tasks:
            self.dispatch_task(task)

    def run(self):
        try:
            self.storage.set("global_controller", "OK")
            while True:
                time.sleep(30)
                self.pause_jobs()
                # do check here
                comms = self.get_communicators()
                topo = GlobalTopo(comms)
                for r in topo.rings:
                    self.validate_ring(r)
                time.sleep(5)
                # resume them
                print("Resume jobs!")
                self.resume_jobs()
        except KeyboardInterrupt:
            print("Stop server running!")
            return


def main():
    parser = argparse.ArgumentParser("Global side controller of fail-slow detection")
    parser.add_argument("-m", "--master_addr", default="127.0.0.1", type=str)
    parser.add_argument("-p", "--port", default=6379, type=int)
    args = parser.parse_args()
    server = GlobalServer(args.master_addr, args.port)
    server.run()


if __name__ == "__main__":
    main()
