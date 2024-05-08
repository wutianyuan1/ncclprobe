#pragma once
#include <chrono>
#include <memory>
#include <iostream>
#include <vector>
#include <exception>
#include <cuda.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include "config.hpp"
#include "shm_storage.hpp"
#include "shm_topo.hpp"

using namespace std::chrono;


struct GlobalStatus {
    // NCCL dynamic linked library
    std::string nccl_lib_path;
    void* nccl_lib_handle;

    // storage buffer on shared memory
    std::shared_ptr<NcclRecordStorage> storage_buffer;

    // Topo links between GPUs
    std::shared_ptr<NcclTopoConnection> topo_buffer;
    std::vector<Communicator> local_comms;
    ncclComm_t comm_in_group;
    uint64_t comm_nccl_id_hash;

    // timing utils
    cudaEvent_t group_op_start, group_op_stop;
    cudaStream_t curr_stream;
    NcclNumber event_op;
    bool has_events_in_group;
    bool in_group;

    // TP related compression operations
    NcclNumber last_call_id;
    uint64_t repeated_call_num;
    uint64_t accumulated_count;
    float accumulated_duration;

    // Running time since training starts
    system_clock::time_point start_time;

    // Temporary record buffer
    std::vector<Record> tmp_record_buffer;

public:
    GlobalStatus() = default;
    GlobalStatus(const char* nccl_path_);

    // Initializes all status
    void initialize(const char* nccl_path_);

    // Returns a function pointer of given `func_name` from NCCL lib
    void* get_function_ptr(const char* func_name);

    // Returns the time since probe initialization in microseconds (us)
    double time_since_initialize();

    // Entry & Exit of NCCL groups
    void group_start();
    void group_end();

    // Updates the TP accumulated calls (AllGather, ReduceScatter)
    void update_accumulation(NcclNumber last_call_number, uint64_t count, float op_duration);

    // Resets the TP accumulated calls (AllGather, ReduceScatter)
    void reset_accumulation(NcclNumber last_call_number);

    // Add a cudaEvent before an NCCL operation for timing 
    void add_timing_event(NcclNumber op, uint64_t count, cudaStream_t stream);

    // Logs this event cost in microseconds (us)
    double get_communication_time();
};