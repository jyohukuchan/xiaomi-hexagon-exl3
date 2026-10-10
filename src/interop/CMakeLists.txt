cmake_minimum_required(VERSION 3.22)
project(exl3_gpu_interop C CXX ASM)
include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
add_library(exl3_gpu_htp SHARED gpu_service.c ${CMAKE_CURRENT_BINARY_DIR}/gpu_iface_skel.c)
target_include_directories(exl3_gpu_htp PRIVATE ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
build_idl(gpu_iface.idl exl3_gpu_htp)
