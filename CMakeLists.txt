cmake_minimum_required (VERSION 3.9)
project(ncclprobe LANGUAGES CXX CUDA)

# required libs: nccl, mpi
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(NCCL REQUIRED)
find_package(MPI REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${NCCL_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${NCCL_LIBRARY_DIRS})
link_directories(${MPI_LIBRARY_PATH})

## The intercept library
add_library(ncclprobe SHARED ncclprobe.cu shm_storage.cpp)
# H100/H800: 90, RTX40*: 89, A10: 86, A100/A30: 80, V100: 70
set_target_properties(ncclprobe PROPERTIES CUDA_ARCHITECTURES "70;80;86;89;90")
target_link_libraries(ncclprobe ${CUDA_LIBRARIES} ${NCCL_LIBRARIES} ${MPI_CXX_LIBRARIES})

## The allreduce test program
add_executable(test test.cu)
set_source_files_properties(test.cu PROPERTIES LANGUAGE CUDA)
# H100/H800: 90, RTX40*: 89, A10: 86, A100/A30: 80, V100: 70
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "70;80;86;89;90")
target_link_libraries(test ${CUDA_LIBRARIES} ${NCCL_LIBRARIES} ${MPI_CXX_LIBRARIES})

