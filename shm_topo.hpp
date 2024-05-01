#pragma once
#include <map>
#include <string>
#include <sstream>
#include <boost/log/trivial.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include "nccl_structs.hpp"


struct Communicator
{
    uint64_t num_channels;
    uint64_t last_ring_id, last_tree_id;
    uint64_t comm_addr;
    int global_rank;
    int local_rank;
    int group_rank;
    ncclRing rings[MAXCHANNELS];
    ncclTree trees[MAXCHANNELS];
    int comm_ops[NCCL_NUM_FUNCTIONS];
public:
    Communicator();
    Communicator(uint64_t addr, int my_rank, uint64_t num_channels);
    ~Communicator();
    void add_ring(ncclRing& ring);
    void add_tree(ncclTree& tree);
    void debug_print();
};


class NcclTopoConnection
{
    uint64_t* n_comms_ptr;
    Communicator* all_comms;
    std::map<uint64_t, int> comm_map;
    boost::interprocess::shared_memory_object shm;
    boost::interprocess::mapped_region region;
public:
    NcclTopoConnection(int n_ranks);
    ~NcclTopoConnection();
    bool add_comm(Communicator& comm);
    Communicator* find(uint64_t comm_addr);
    uint64_t num_comms() const;
};
