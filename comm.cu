#include "comm.hpp"

// pure reverse engineering, I'm sb
void get_comm(ncclComm_t comm, std::shared_ptr<NcclTopoConnection> topo)
{
    HackedComm* hcomm = reinterpret_cast<HackedComm*>(comm);

    Communicator* ret = topo->find(reinterpret_cast<uint64_t>(comm), hcomm->rank);
    printf("!! add comm, rank=%d, addr=%p, comms in buffer: %d\n", hcomm->rank, comm, topo->num_comms());
    if (ret) {
        printf("!!! comm %p is found!\n", ret);
        return;
    }

    Communicator my_comm(
        reinterpret_cast<uint64_t>(hcomm), hcomm->rank, hcomm->nChannels);
    for (int i = 0; i < hcomm->nChannels; i++){
      my_comm.add_ring(hcomm->channels[i].ring);
      my_comm.add_tree(hcomm->channels[i].tree);
    }
    printf("!!! Add comm %p!\n", comm);
    my_comm.debug_print();
    topo->add_comm(my_comm);
}