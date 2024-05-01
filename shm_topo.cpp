#include "shm_topo.hpp"

using namespace boost::interprocess;


Communicator::Communicator(uint64_t addr, int my_rank, uint64_t num_channels_) 
    : num_channels(num_channels_), last_ring_id(0), last_tree_id(0), comm_addr(addr), global_rank(my_rank)
{}

Communicator::Communicator() 
    : num_channels(0), last_ring_id(0), last_tree_id(0), comm_addr(0), global_rank(0)
{}

Communicator::~Communicator()
{}

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
        << "  Global Rank:" << global_rank << ", Group Rank: " << group_rank << ", Local Rank: " << local_rank << ", #channels: " << num_channels;
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


bool NcclTopoConnection::add_comm(Communicator& comm)
{
    if (this->find(comm.comm_addr))
    {
        BOOST_LOG_TRIVIAL(info) << "Communicator " << comm.comm_addr << "is found in cache, will not be repeatly added";
        return false;
    }
    memcpy(&all_comms[*n_comms_ptr], &comm, sizeof(Communicator));
    memcpy(all_comms[*n_comms_ptr].rings, comm.rings, sizeof(ncclRing) * comm.num_channels);
    memcpy(all_comms[*n_comms_ptr].trees, comm.trees, sizeof(ncclTree) * comm.num_channels);
    comm_map.insert({comm.comm_addr, *n_comms_ptr});
    (*n_comms_ptr)++;
    return true;
}

Communicator* NcclTopoConnection::find(uint64_t comm_addr)
{
    auto iter = comm_map.find(comm_addr);
    return iter == comm_map.end() ? nullptr : &all_comms[iter->second];
}

uint64_t NcclTopoConnection::num_comms() const
{
    return (*n_comms_ptr);
}