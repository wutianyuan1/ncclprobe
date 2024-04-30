#include "shm_topo.hpp"

using namespace boost::interprocess;


uint64_t hash_comm(uint64_t addr, int rank)
{
    uint64_t hashcode = 23;
    hashcode = (hashcode * 37) + addr;
    hashcode = (hashcode * 37) + rank;
    return hashcode;
}


Communicator::Communicator(uint64_t addr, int my_rank, uint64_t num_channels_) 
    : num_channels(num_channels_), last_ring_id(0), last_tree_id(0), comm_addr(addr), rank(my_rank)
{
    this->rings = new ncclRing[num_channels];
    this->trees = new ncclTree[num_channels];
}

Communicator::~Communicator()
{
    if (rings) delete[] rings;
    if (trees) delete[] trees;
}

void Communicator::add_ring(ncclRing& ring)
{
    memcpy(rings + last_ring_id, &ring, sizeof(struct ncclRing));
    last_ring_id++;
}

void Communicator::add_tree(ncclTree& tree)
{
    memcpy(trees + last_tree_id, &tree, sizeof(struct ncclTree));
    last_tree_id++;
}

void Communicator::debug_print()
{
    std::stringstream ss;
    ss << "<GPU Connection Info>\n"
        << "  Rank:" << rank << ", #channels: " << num_channels;
    for (int i = 0; i < num_channels; i++)
        ss << "  channel[" << i << "]:\n"
            << "    (Ring id=" << rings[i].index << ", prev=" << rings[i].prev << ", next=" << rings[i].next << ")\n"
            << "    (Tree depth=" << trees[i].depth << ", up=" << trees[i].up << ", down=(" << trees[i].down[0] << ", " << trees[i].down[1] <<"))\n";
    BOOST_LOG_TRIVIAL(info) << ss.str();
}


NcclTopoConnection::NcclTopoConnection(int n_ranks)
{
    shm = shared_memory_object(open_or_create, "ncclTopo", read_write);
    shm.truncate(n_ranks * MAX_COMMS_PER_RANK * sizeof(Communicator));
    region = mapped_region(shm, read_write);
    unsigned char* shm_addr = reinterpret_cast<unsigned char*>(region.get_address());
    n_comms_ptr = reinterpret_cast<uint64_t*>(shm_addr);
    all_comms = reinterpret_cast<Communicator*>(shm_addr + sizeof(uint64_t));
}

NcclTopoConnection::~NcclTopoConnection()
{
    shared_memory_object::remove("ncclTopo");
}


void NcclTopoConnection::add_comm(Communicator& comm)
{
    uint64_t hash = hash_comm(comm.comm_addr, comm.rank);
    memcpy(&all_comms[*n_comms_ptr], &comm, sizeof(Communicator));
    comm_map.insert({hash, *n_comms_ptr});
    (*n_comms_ptr)++;
}

Communicator* NcclTopoConnection::find(uint64_t comm_addr, int rank)
{
    uint64_t hash = hash_comm(comm_addr, rank);
    auto iter = comm_map.find(hash);
    return iter == comm_map.end() ? nullptr : &all_comms[iter->second];
}

uint64_t NcclTopoConnection::num_comms() const
{
    return (*n_comms_ptr);
}