from typing import List, Tuple


def split_ring(ring: List[int]) -> List[Tuple[List[int]]]:
    if len(ring) == 2:
        first_send = [ring[0]]
        first_recv = [ring[1]]
        second_send = [ring[1]]
        second_recv = [ring[0]]
        return [(first_send, first_recv), (second_send, second_recv)]
    if len(ring) % 2 == 0:
        first_send = ring[0::2]
        first_recv = ring[1::2]
        second_send = ring[1::2]
        second_recv = ring[2:-1:2] + [ring[0]]
        return [(first_send, first_recv), (second_send, second_recv)]
    else:
        first_send = ring[0:-1:2]
        first_recv = ring[1::2]
        second_send = ring[1::2]
        second_recv = ring[2::2]
        third_send = [ring[-1]]
        third_recv = [ring[0]]
        return [(first_send, first_recv), (second_send, second_recv), (third_send, third_recv)]
