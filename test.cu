#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>


#define N_REPEAT 10

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static uint64_t getIDHash(const char* ID) {
  return getHostHash(ID);
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


__global__ void fillBuffer(float *buf, int value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = value;
    }
}



int main(int argc, char* argv[])
{
  int size = 16;
  int myRank, nRanks, localRank = 0;

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }
  printf("[Rank %d] localRank=%d\n", myRank, localRank);

  // Define groups
  int group = myRank / 2; // This will be 0 for ranks 0,1 and 1 for ranks 2,3

  // Create MPI sub-communicators based on groups
  MPI_Comm subComm;
  MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, group, myRank, &subComm));

  int subRank, subSize;
  MPICHECK(MPI_Comm_rank(subComm, &subRank));
  MPICHECK(MPI_Comm_size(subComm, &subSize));

  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  // Get NCCL unique ID at sub-communicator rank 0 and broadcast it to all others in the sub-communicator
  if (subRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, subComm));
  printf("[Rank %d], subRank=%d, subSize=%d, unique ID=%lu\n", myRank, subRank, subSize, getIDHash(id.internal));

  // Picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  fillBuffer<<<blocksPerGrid, threadsPerBlock, 0, s>>>(sendbuff, myRank, size);
  CUDACHECK(cudaStreamSynchronize(s));

  // Initializing NCCL with sub-communicator
  NCCLCHECK(ncclCommInitRank(&comm, subSize, id, subRank));

  // Communicating using NCCL within sub-communicators
  for (int i = 0; i < N_REPEAT; i++) {
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclAvg, comm, s));
    NCCLCHECK(ncclGroupEnd());    
  }

  // Completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  float* cpubuff = (float*)malloc(sizeof(float) * size);
  cudaMemcpy(cpubuff, recvbuff, sizeof(float)*size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("[Rank %d]: result=%f\n", myRank, cpubuff[0]);

  // Free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  // Finalizing NCCL
  ncclCommDestroy(comm);

  // Finalizing MPI sub-communicator
  MPICHECK(MPI_Comm_free(&subComm));

  // Finalizing MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
