cmake_minimum_required(VERSION 3.22)
project(attendance_system)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── stdc++fs for GCC < 9 ────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
        link_libraries(stdc++fs)
    endif()
endif()

# ── OpenVINO path (auto-detected from Python env) ────────────────────────────
find_package(Python3 REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
    OUTPUT_VARIABLE OpenVINO_DIR_PY
    ERROR_QUIET
)

# ── Fake CUDA setup to satisfy OpenCV built with CUDA ────────────────────────
set(FAKE_CUDA_DIR "${CMAKE_BINARY_DIR}/fake_cuda")
file(MAKE_DIRECTORY "${FAKE_CUDA_DIR}/bin")
file(MAKE_DIRECTORY "${FAKE_CUDA_DIR}/include")
file(MAKE_DIRECTORY "${FAKE_CUDA_DIR}/lib64")
file(WRITE "${FAKE_CUDA_DIR}/include/cuda.h"     "/* fake */\n")
file(WRITE "${FAKE_CUDA_DIR}/lib64/libcudart.so" "")
file(WRITE "${FAKE_CUDA_DIR}/bin/nvcc"
    "#!/bin/sh\nprintf 'Cuda compilation tools, release 12.2, V12.2.0\\n'")
execute_process(COMMAND chmod +x "${FAKE_CUDA_DIR}/bin/nvcc")

set(CUDA_TOOLKIT_ROOT_DIR "${FAKE_CUDA_DIR}"          CACHE PATH     "" FORCE)
set(CUDA_NVCC_EXECUTABLE  "${FAKE_CUDA_DIR}/bin/nvcc" CACHE FILEPATH "" FORCE)
include(FindCUDA)
set(CUDA_FOUND           TRUE                                   CACHE BOOL     "" FORCE)
set(CUDA_VERSION         "12.2"                                 CACHE STRING   "" FORCE)
set(CUDA_VERSION_STRING  "12.2"                                 CACHE STRING   "" FORCE)
set(CUDA_VERSION_MAJOR   "12"                                   CACHE STRING   "" FORCE)
set(CUDA_VERSION_MINOR   "2"                                    CACHE STRING   "" FORCE)
set(CUDA_INCLUDE_DIRS    "${FAKE_CUDA_DIR}/include"             CACHE PATH     "" FORCE)
set(CUDA_CUDART_LIBRARY  "${FAKE_CUDA_DIR}/lib64/libcudart.so"  CACHE FILEPATH "" FORCE)
set(CUDA_LIBRARIES       "${FAKE_CUDA_DIR}/lib64/libcudart.so"  CACHE STRING   "" FORCE)
set(CUDA_npp_LIBRARY     "" CACHE FILEPATH "" FORCE)
set(CUDA_nppc_LIBRARY    "" CACHE FILEPATH "" FORCE)
set(CUDA_nppi_LIBRARY    "" CACHE FILEPATH "" FORCE)
set(CUDA_npps_LIBRARY    "" CACHE FILEPATH "" FORCE)
set(CUDA_CUBLAS_LIBRARIES "" CACHE STRING  "" FORCE)
set(CUDA_CUFFT_LIBRARIES  "" CACHE STRING  "" FORCE)

# ── Third-party packages ─────────────────────────────────────────────────────
find_package(OpenCV   REQUIRED)
find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")
find_package(OpenSSL  REQUIRED)

# ── Project source layout ────────────────────────────────────────────────────
#   src/        — attendance system sources (main, recognizer, db, logger)
#   vendor/src/ — base face-recognition engine sources
#   include/    — attendance system headers (patched, must come first)
#   vendor/include/ — base engine headers

set(VENDOR_SRC_DIR "${CMAKE_SOURCE_DIR}/vendor/src")
set(VENDOR_INC_DIR "${CMAKE_SOURCE_DIR}/vendor/include")

# ── Source files ─────────────────────────────────────────────────────────────
file(GLOB ATTENDANCE_SOURCES
    src/main.cpp
    src/face_database.cpp
    src/attendance_logger.cpp
    src/recognizer.cpp
)

set(VENDOR_SOURCES
    ${VENDOR_SRC_DIR}/arcface.cpp
    ${VENDOR_SRC_DIR}/facenet.cpp
    ${VENDOR_SRC_DIR}/face_embedding.cpp
    ${VENDOR_SRC_DIR}/inference.cpp
    ${VENDOR_SRC_DIR}/preprocessing.cpp
    ${VENDOR_SRC_DIR}/model_decryptor.cpp
)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME}
    ${ATTENDANCE_SOURCES}
    ${VENDOR_SOURCES}
)

# ── Include directories ──────────────────────────────────────────────────────
target_include_directories(${PROJECT_NAME} PRIVATE
    include              # MUST be first — patched model_selection.hpp etc.
    ${VENDOR_INC_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

# ── Link libraries ───────────────────────────────────────────────────────────
target_link_libraries(${PROJECT_NAME} PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
    OpenSSL::SSL
    OpenSSL::Crypto
)

# ── RPATH ────────────────────────────────────────────────────────────────────
set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
