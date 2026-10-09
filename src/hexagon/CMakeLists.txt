cmake_minimum_required(VERSION 3.22)
project(exl3_hexagon C CXX ASM)
include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
add_library(exl3_htp SHARED exl3_service.c ${CMAKE_CURRENT_BINARY_DIR}/exl3_iface_skel.c)
target_include_directories(exl3_htp PRIVATE ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef ${CMAKE_CURRENT_BINARY_DIR})
build_idl(exl3_iface.idl exl3_htp)
