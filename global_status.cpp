#include "global_status.hpp"
#include "utils.hpp"

#ifndef RECORD_TOO_SMALL
    #define RECORD_TOO_SMALL(count) ((count) < MIN_RECORD_OP_SIZE)
#endif


GlobalStatus::GlobalStatus(const char* nccl_path_)
    : nccl_lib_handle(nullptr), storage_buffer(nullptr)
{
    initialize(nccl_path_);
}

GlobalStatus::~GlobalStatus()
{
    if (global_controller_proc != nullptr)
        global_controller_proc->terminate();
    if (local_controller_proc != nullptr)
        local_controller_proc->terminate();
}

void GlobalStatus::initialize(const char* nccl_path_)
{
    // Initialize logging system
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
    );

    // start the global controller if it is the master rank (rank 0)
    if (get_rank(DistEngine::auto_find) == 0)
        start_global_controller();
    // start the local controller if it is the local master rank (local rank 0)
    if (get_local_rank(DistEngine::auto_find) == 0)
        start_local_controller();
    sleep(3);

    // First, find & load the NCCL dynamic lib
    if (nccl_path_ == nullptr)
        this->nccl_lib_path = std::string("/usr/lib/x86_64-linux-gnu/libnccl.so.2");
    else
        this->nccl_lib_path = std::string(nccl_path_);
    void* handle = dlopen(this->nccl_lib_path.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
        std::cerr << "Error: " << dlerror() << std::endl;
        throw std::runtime_error(dlerror());
    }
    dlerror();
    this->nccl_lib_handle = handle;

    // Second, create the shared memory storage buffer and topological buffer
    this->storage_buffer = std::shared_ptr<NcclRecordStorage>(
        new NcclRecordStorage(Record::numFields(), BUFFER_SIZE));

    this->topo_buffer =  std::shared_ptr<NcclTopoConnection>(
        new NcclTopoConnection(4)  //!!replace to correct n_ranks
    );
    this->comm_in_group = nullptr;
    this->local_comms.clear();

    if (get_rank(DistEngine::auto_find) == 0)
        BOOST_LOG_TRIVIAL(debug) << "The buffer contains " << Record::numFields() << " fields." << std::endl;

    // Third, initialize event timing utils
    this->group_op_start = nullptr;
    this->group_op_stop = nullptr;
    this->curr_stream = nullptr;
    this->event_op = NcclNumber::INVALID;
    this->has_events_in_group = false;
    this->in_group = false;
    cudaEventCreate(&group_op_start);
    cudaEventCreate(&group_op_stop);

    // Then, initialize the TP related variables
    this->last_call_id = NcclNumber::INVALID;
    this->last_comm = nullptr;
    this->repeated_call_num = 0;
    this->accumulated_count = 0;
    this->accumulated_duration = 0.0;

    // Initialize interaction compoent
    this->should_check = false;
    this->event_handler = std::shared_ptr<EventHandler>(
        new EventHandler("127.0.0.1", 6379)
    );

    // Set the running state to "MONITOR"
    this->state = ControlState::STATE_MONITOR;

    // Finally, initialize the start running time
    start_time = system_clock::now();

    BOOST_LOG_TRIVIAL(info) << "Rank: " << get_rank(DistEngine::auto_find) << ", global Status initialized!!" << std::endl;
}

int GlobalStatus::start_global_controller()
{
    namespace bp = boost::process;
    std::vector<std::string> args {
        "-c", "python /workspace/ncclprobe/control_plane/global_controller.py"
    };
    global_controller_proc = std::shared_ptr<bp::child>(new bp::child(bp::search_path("sh"), args));
    BOOST_LOG_TRIVIAL(info) << "[Master rank] Global controller started";
    return 0;
}

int GlobalStatus::start_local_controller()
{
    namespace bp = boost::process;
    std::vector<std::string> args {
        "-c", "python /workspace/ncclprobe/control_plane/local_controller.py"
    };
    local_controller_proc = std::shared_ptr<bp::child>(new bp::child(bp::search_path("sh"), args));
    BOOST_LOG_TRIVIAL(info) << "[Local Master rank] Local controller started";
    return 0;
}

void GlobalStatus::reset_accumulation(NcclNumber last_call_number)
{
    repeated_call_num = 0;
    accumulated_count = 0;
    accumulated_duration = 0.0;
    last_comm = nullptr;
    last_call_id = last_call_number;
}

void GlobalStatus::update_accumulation(NcclNumber last_call_number, uint64_t count, ncclComm_t tp_comm)
{
    repeated_call_num++;
    last_call_id = last_call_number;
    last_comm = tp_comm;
    accumulated_count += count;
}

void* GlobalStatus::get_function_ptr(const char* func_name)
{
    return dlsym(nccl_lib_handle, func_name);
}

double GlobalStatus::time_since_initialize()
{
    return (double)(duration_cast<microseconds>(system_clock::now() - start_time).count());
}


void GlobalStatus::group_start()
{
    in_group = true;
    has_events_in_group = false;
    curr_stream = nullptr;
}


void GlobalStatus::group_end()
{
    in_group = false;
    has_events_in_group = false;
    curr_stream = nullptr;
}


void GlobalStatus::add_timing_event(NcclNumber op, uint64_t count, cudaStream_t stream)
{
    // Do nothing if we are in the monitoring state
    if (state == ControlState::STATE_MONITOR)
        return;
    // If the operation size<1K, skip it
    if (RECORD_TOO_SMALL(count))
        return;
    // Else, record it
    event_op = op;
    // The CUDA event will be added if:
    // (1) We are in a NCCL group but it is the first call; (2) We are not in group
    if (!has_events_in_group || !in_group) {
        if (in_group)
            has_events_in_group = true;
        cudaEventRecord(group_op_start, stream);
        curr_stream = stream;
    }
}

double GlobalStatus::get_communication_time()
{
    // skip if we are in the monitoring state
    if (state == ControlState::STATE_MONITOR)
        return 0.0;
    // If no operations are recorded (i.e., we are in a group but no calls within it), return 0.0
    if (in_group && !has_events_in_group)
        return 0.0;
    // Else report the event time
    float duration = 0;
    cudaEventRecord(group_op_stop, curr_stream);
    cudaEventSynchronize(group_op_stop);
    cudaEventElapsedTime(&duration, group_op_start, group_op_stop);
    // convert ms to us
    return duration * 1000.0;
}
