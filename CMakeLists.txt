cmake_minimum_required(VERSION 3.22)
project(RPPLabs)

set(CMAKE_CXX_STANDARD 14)

add_executable(RPPLabs openmp.c mpi.c)
target_link_libraries(RPPLabs PRIVATE OpenMP::OpenMP_C)
