from typing import List, Dict
from communicator import Communicator, RingNode, TreeNode


def ring_key(ring: List[int]):
    return tuple(sorted(ring))


class GlobalTopo(object):
    def __init__(self, comms: List[Communicator]) -> None:
        self.comms = {c.comm_addr : c for c in comms}
        self.comms_by_id_hash = {}
        for comm in self.comms.values():
            if comm.id_hash in self.comms_by_id_hash:
                self.comms_by_id_hash[comm.id_hash].append(comm)
            else:
                self.comms_by_id_hash[comm.id_hash] = [comm]
        self.rings = self.build_rings(self.comms_by_id_hash)
        self.trees = self.build_trees(self.comms_by_id_hash)

    def build_rings(self, comms_by_id_hash: Dict[int, List[Communicator]]):
        rings = []
        for (id_hash, comms) in comms_by_id_hash.items():
            comms_by_rank = {cm.group_rank: cm for cm in comms}
            # for ch in range(comms[0].num_channels):
            # Now always use the 0th channel because all channels are identical
            curr_ring = []
            visited = set()
            curr_node: RingNode = comms[0].rings[0]
            while len(visited) != len(comms):
                curr_ring.append(curr_node.index)
                visited.add(curr_node.index)
                curr_node = comms_by_rank[curr_node.next].rings[0]
            rings.append(curr_ring)
        
        unique_rings = {}
        final_rings = []
        for r in rings:
            ring_hash = ring_key(r)
            if ring_hash not in unique_rings:
                unique_rings[ring_hash] = True
                final_rings.append(r)
        return final_rings
        

    def build_trees(self, comms_by_id_hash: Dict[int, List[Communicator]]):
        rings = []
        for (id_hash, comms) in comms_by_id_hash.items():
            pass
