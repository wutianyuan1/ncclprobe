#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include <unistd.h>
#include <memory>
#include "shm_storage.hpp"
#include "config.hpp"


using namespace std::chrono;
static bool probe_inited = false;
static void* nccl_lib_handle = nullptr;
static auto start_time = system_clock::now();
std::shared_ptr<NcclRecordStorage> storage_buffer = nullptr;


bool init_probe(){
    char* ncclPath = getenv("NCCL_PATH");
    if (!ncclPath) ncclPath = (char*)"/usr/lib/x86_64-linux-gnu/libnccl.so.2";
    void* handle = dlopen(ncclPath, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return false;
    }
    dlerror();
    nccl_lib_handle = handle;
    probe_inited = true;
    std::cout << Record::numFields() << std::endl;
    storage_buffer = std::shared_ptr<NcclRecordStorage>(
        new NcclRecordStorage(Record::numFields(), BUFFER_SIZE));
    return true;
}


ncclResult_t probe_begin(const void* buff1, const void* buff2, size_t count,
                        ncclDataType_t datatype, ncclComm_t comm,
                        cudaStream_t stream, NcclNumber number, uint64_t aux)
{
    int dev_id = -1, caller = -1, numdevs = -1;
    char pcistr[PCI_STR_LEN] = {0};
    auto call_time = (double)(duration_cast<milliseconds>(system_clock::now() - start_time).count()); 
    uint64_t comm_devices[MAX_DEVS];

    memset(comm_devices, 0, sizeof(comm_devices));
    cudaGetDevice(&dev_id);
    cudaDeviceGetPCIBusId(pcistr, PCI_STR_LEN, dev_id);
    ncclCommUserRank(comm, &caller);
    ncclCommCount(comm, &numdevs);

    Record record(
        (uint64_t)number, count, reinterpret_cast<uint64_t>(buff1),
        reinterpret_cast<uint64_t>(buff2), (uint64_t)(datatype),
        (uint64_t)(getpid()), (uint64_t)(call_time * 1000), (uint64_t)(dev_id),
        (uint64_t)(caller), aux, (uint64_t)(numdevs), comm_devices
    );
    
    storage_buffer->addRecord(record.toVector());
    return ncclSuccess;
}

ncclResult_t probe_end()
{
    return ncclSuccess;
}

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream)
{
    if (!probe_inited)
        if (!init_probe()) exit(1);
    using func_t = typeof(ncclSend);
    auto real_func = reinterpret_cast<func_t*>(dlsym(nccl_lib_handle, "ncclSend"));
    probe_begin(sendbuff, nullptr, count, datatype, comm, stream, NcclNumber::SEND, (uint64_t)peer);
    auto ret = (*real_func)(sendbuff, count, datatype, peer, comm, stream);
    probe_end();
    return ret;
}


ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream)
{
    if (!probe_inited)
        if (!init_probe()) exit(1);
    using func_t = typeof(ncclRecv);
    auto real_func = reinterpret_cast<func_t*>(dlsym(nccl_lib_handle, "ncclRecv"));
    probe_begin(nullptr, recvbuff, count, datatype, comm, stream, NcclNumber::RECV, (uint64_t)peer);
    auto ret = (*real_func)(recvbuff, count, datatype, peer, comm, stream);
    probe_end();
    return ret;
}


ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype,
                      int root, ncclComm_t comm, cudaStream_t stream)
{
    if (!probe_inited)
        if (!init_probe()) exit(1);
    using func_t = typeof(ncclBcast);
    auto real_func = reinterpret_cast<func_t*>(dlsym(nccl_lib_handle, "ncclBcast"));
    probe_begin(buff, nullptr, count, datatype, comm, stream, NcclNumber::BCAST, (uint64_t)root);
    auto ret = (*real_func)(buff, count, datatype, root, comm, stream);
    probe_end();
    return ret;
}


ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream)
{
    if (!probe_inited)
        if (!init_probe()) exit(1);
    using func_t = typeof(ncclBroadcast);
    auto real_func = reinterpret_cast<func_t*>(dlsym(nccl_lib_handle, "ncclBroadcast"));
    probe_begin(sendbuff, recvbuff, count, datatype, comm, stream, NcclNumber::BROADCAST, (uint64_t)root);
    auto ret = (*real_func)(sendbuff, recvbuff, count, datatype, root, comm, stream);
    probe_end();
    return ret;
}


ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
{
    if (!probe_inited)
        if (!init_probe()) exit(1);
    using func_t = typeof(ncclAllGather);
    auto real_func = reinterpret_cast<func_t*>(dlsym(nccl_lib_handle, "ncclAllGather"));
    probe_begin(sendbuff, recvbuff, sendcount, datatype, comm, stream, NcclNumber::ALL_GATHER, (uint64_t)0);
    auto ret = (*real_func)(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    probe_end();
    return ret;
}


ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                               ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream)
{
    if (!probe_inited)
        if (!init_probe()) exit(1);
    using func_t = typeof(ncclReduceScatter);
    auto real_func = reinterpret_cast<func_t*>(dlsym(nccl_lib_handle, "ncclReduceScatter"));
    probe_begin(sendbuff, recvbuff, recvcount, datatype, comm, stream, NcclNumber::REDUCE_SCATTER, (uint64_t)op);
    auto ret = (*real_func)(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    probe_end();
    return ret;
}


ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream)
{
    if (!probe_inited)
        if (!init_probe()) exit(1);
    using func_t = typeof(ncclAllReduce);
    auto real_func = reinterpret_cast<func_t*>(dlsym(nccl_lib_handle, "ncclAllReduce"));
    probe_begin(sendbuff, recvbuff, count, datatype, comm, stream, NcclNumber::ALL_REDUCE, (uint64_t)op);
    auto ret = (*real_func)(sendbuff, recvbuff, count, datatype, op, comm, stream);
    probe_end();
    return ret;
}
