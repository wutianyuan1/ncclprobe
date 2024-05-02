cmake_minimum_required (VERSION 3.9)
project(ncclprobe LANGUAGES CXX CUDA)

SET(CMAKE_BUILD_TYPE "DEBUG")
enable_language(CUDA)
set_source_files_properties(*.cpp PROPERTIES LANGUAGE CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# required libs: nccl, mpi
find_package(NCCL REQUIRED)
find_package(MPI REQUIRED)

# boost log
find_package(Boost COMPONENTS log REQUIRED)

# cpp_redis
find_package(CPPREDIS REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${NCCL_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})
include_directories(${CPPREDIS_INCLUDE_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${NCCL_LIBRARY_DIRS})
link_directories(${MPI_LIBRARY_PATH})
link_directories(${CPPREDIS_LIBRARY})
link_directories(${TACOPIE_LIBRARY})


## The intercept library
add_library(ncclprobe SHARED ncclprobe.cu shm_storage.cpp global_status.cu comm.cu shm_topo.cpp)
# H100/H800: 90, RTX40*: 89, A10: 86, A100/A30: 80, V100: 70
set_target_properties(ncclprobe PROPERTIES CUDA_ARCHITECTURES "70;80;86;89;90")
target_link_libraries(ncclprobe ${CUDA_LIBRARIES} ${MPI_CXX_LIBRARIES} ${CPPREDIS_LIBRARY} ${TACOPIE_LIBRARY} Boost::log)

## The allreduce test program
add_executable(test test.cu)
set_source_files_properties(test.cu PROPERTIES LANGUAGE CUDA)
# H100/H800: 90, RTX40*: 89, A10: 86, A100/A30: 80, V100: 70
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "70;80;86;89;90")
target_link_libraries(test ${CUDA_LIBRARIES} ${NCCL_LIBRARIES} ${MPI_CXX_LIBRARIES})
