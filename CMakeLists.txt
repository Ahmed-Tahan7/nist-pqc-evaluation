cmake_minimum_required(VERSION 3.20)
project(pqc_fault_analysis C)

set(CMAKE_C_STANDARD 11)

set(LIBOQS_PREFIX "/opt/homebrew" CACHE PATH "liboqs install prefix")

include_directories(${LIBOQS_PREFIX}/include)
link_directories(${LIBOQS_PREFIX}/lib)

find_package(OpenSSL REQUIRED)

add_executable(pqc_fault_analysis
        bench_main.c
        bench_mlkem.c
        bench_mldsa.c
        bench_slhdsa.c
)

target_link_libraries(pqc_fault_analysis
        oqs
        OpenSSL::Crypto
)

if(APPLE)
        set_target_properties(pqc_fault_analysis PROPERTIES
                BUILD_RPATH "${LIBOQS_PREFIX}/lib"
                INSTALL_RPATH "${LIBOQS_PREFIX}/lib"
        )
endif()