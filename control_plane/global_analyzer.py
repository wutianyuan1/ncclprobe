import redis
import time
from communicator import Communicator
from typing import List, Dict


class GlobalAnalyzer(object):
    def __init__(self, storage: redis.StrictRedis):
        self.storage = storage
        self.world_size = None
        while self.world_size is None:
            self.world_size = self.storage.get("world_size")
            time.sleep(1)
        self.world_size = int(self.world_size.decode())
        self.tp, self.dp, self.pp = self.get_parallel_states()
        self.comms = None

    def build_comm_cliques(self, comms: List[Communicator]):
        idhash2clique: Dict[int, List[Communicator]] = {}
        for c in comms:
            if c.id_hash not in idhash2clique:
                idhash2clique[c.id_hash] = [c]
            else:
                idhash2clique[c.id_hash].append(c)
        addr2clique = {}
        for comm_clique in idhash2clique.values():
            for c in comm_clique:
                addr2clique[c.comm_addr] = comm_clique
        return addr2clique
    
    def get_parallel_states(self):
        tp, dp, pp = [], [], []
        parse_rankstr = lambda s: [int(i) for i in s.split("_")]
        for rank in range(self.world_size):
            tp_r = parse_rankstr(self.storage.get(f"{rank}_tp").decode())
            dp_r = parse_rankstr(self.storage.get(f"{rank}_dp").decode())
            pp_r = parse_rankstr(self.storage.get(f"{rank}_pp").decode())
            tp.append(tuple(tp_r))
            dp.append(tuple(dp_r))
            pp.append(tuple(pp_r))
        # remove duplicates
        tp = list(set(tp))
        dp = list(set(dp))
        pp = list(set(pp))
        return [list(i) for i in tp], [list(i) for i in dp], [list(i) for i in pp]
