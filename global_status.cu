#include "global_status.hpp"
#ifndef RECORD_TOO_SMALL
    #define RECORD_TOO_SMALL(count) ((count) < MIN_RECORD_OP_SIZE)
#endif

GlobalStatus::GlobalStatus(const char* nccl_path_)
    : nccl_lib_handle(nullptr), storage_buffer(nullptr)
{
    initialize(nccl_path_);
}

void GlobalStatus::initialize(const char* nccl_path_)
{
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

    // Second, create the shared memory storage buffer
    this->storage_buffer = std::shared_ptr<NcclRecordStorage>(
        new NcclRecordStorage(Record::numFields(), BUFFER_SIZE));
    if (std::atoi(getenv("RANK")) == 0)
        std::cout << "The buffer contains " << Record::numFields() << " fields." << std::endl;

    // Third, initialize event timing utils
    this->group_op_start = nullptr;
    this->group_op_stop = nullptr;
    this->curr_stream = nullptr;
    this->event_op = NcclNumber::INVALID;
    this->has_events_in_group = false;
    cudaEventCreate(&group_op_start);
    cudaEventCreate(&group_op_stop);

    // Then, initialize the TP related variables
    this->last_call_id = NcclNumber::INVALID;
    this->repeated_call_num = 0;
    this->accumulated_count = 0;
    this->accumulated_duration = 0.0;

    // Finally, initialize the start running time
    start_time = system_clock::now();

    std::cout << "Global Status initialized!!" << std::endl;
}

void GlobalStatus::reset_accumulation(NcclNumber last_call_number)
{
    repeated_call_num = 0;
    accumulated_count = 0;
    accumulated_duration = 0.0;
    last_call_id = last_call_number;
}

void GlobalStatus::update_accumulation(NcclNumber last_call_number, uint64_t count, float duration)
{
    repeated_call_num++;
    last_call_id = last_call_number;
    accumulated_count += count;
    accumulated_duration += duration;
}

void* GlobalStatus::get_function_ptr(const char* func_name)
{
    return dlsym(nccl_lib_handle, func_name);
}

double GlobalStatus::time_since_initialize()
{
    return (double)(duration_cast<microseconds>(system_clock::now() - start_time).count());
}


void GlobalStatus::reset_group_events()
{
    has_events_in_group = false;
    curr_stream = nullptr;
}

void GlobalStatus::add_timing_event(NcclNumber op, uint64_t count, cudaStream_t stream)
{
    // If the operation size<1K, skip it
    if (RECORD_TOO_SMALL(count))
        return;
    // Else, record it
    has_events_in_group = true;
    event_op = op;
    cudaEventRecord(group_op_start, stream);
    curr_stream = stream;   
}

double GlobalStatus::get_communication_time()
{
    // If no operations are recorded, return -1
    if (!has_events_in_group)
        return -1.0;
    // Else report the event time
    float duration = 0;
    cudaEventRecord(group_op_stop, curr_stream);
    cudaEventSynchronize(group_op_stop);
    cudaEventElapsedTime(&duration, group_op_start, group_op_stop);
    // convert ms to us
    return duration * 1000.0;
}
