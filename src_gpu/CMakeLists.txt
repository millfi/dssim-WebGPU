cmake_minimum_required(VERSION 3.24)

add_executable(dssim_gpu_dummy
    dummy_main.cpp
)

target_compile_features(dssim_gpu_dummy PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(dssim_gpu_dummy PRIVATE /W4 /EHsc)
else()
    target_compile_options(dssim_gpu_dummy PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(DSSIM_ENABLE_DAWN_SAMPLE "Build Dawn native compute sample target" OFF)
set(DSSIM_DAWN_ROOT "${CMAKE_SOURCE_DIR}/third_party/dawn" CACHE PATH "Path to Dawn source root")
set(DSSIM_DAWN_OUT_DIR "${DSSIM_DAWN_ROOT}/out/Release" CACHE PATH "Path to Dawn GN build output directory")

if(DSSIM_ENABLE_DAWN_SAMPLE)
    find_path(DSSIM_DAWN_INCLUDE_DIR
        NAMES dawn/webgpu_cpp.h
        PATHS
            "${DSSIM_DAWN_OUT_DIR}/gen/include"
            "${DSSIM_DAWN_ROOT}/include"
        NO_DEFAULT_PATH
    )

    find_path(DSSIM_DAWN_SRC_INCLUDE_DIR
        NAMES dawn/native/DawnNative.h
        PATHS
            "${DSSIM_DAWN_ROOT}/include"
            "${DSSIM_DAWN_ROOT}/src"
        NO_DEFAULT_PATH
    )

    find_library(DSSIM_DAWN_WEBGPU_DAWN_LIB
        NAMES webgpu_dawn webgpu_dawn.dll.lib
        PATHS "${DSSIM_DAWN_OUT_DIR}"
        NO_DEFAULT_PATH
    )

    find_library(DSSIM_DAWN_DAWN_PROC_LIB
        NAMES dawn_proc dawn_proc.dll.lib
        PATHS "${DSSIM_DAWN_OUT_DIR}"
        NO_DEFAULT_PATH
    )

    find_library(DSSIM_DAWN_DAWN_NATIVE_LIB
        NAMES dawn_native dawn_native.dll.lib
        PATHS "${DSSIM_DAWN_OUT_DIR}"
        NO_DEFAULT_PATH
    )

    if(
        NOT DSSIM_DAWN_INCLUDE_DIR OR
        NOT DSSIM_DAWN_SRC_INCLUDE_DIR OR
        NOT DSSIM_DAWN_WEBGPU_DAWN_LIB OR
        NOT DSSIM_DAWN_DAWN_PROC_LIB OR
        NOT DSSIM_DAWN_DAWN_NATIVE_LIB
    )
        message(FATAL_ERROR
            "Dawn sample requested but required headers/libs not found. "
            "Set DSSIM_DAWN_ROOT and DSSIM_DAWN_OUT_DIR correctly."
        )
    endif()

    add_executable(dssim_gpu_dawn_checksum
        dawn_checksum.cpp
    )
    set(DSSIM_GPU_STAGE0_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/stage0_absdiff.wgsl")

    target_compile_features(dssim_gpu_dawn_checksum PRIVATE cxx_std_17)
    target_include_directories(dssim_gpu_dawn_checksum PRIVATE
        "${DSSIM_DAWN_INCLUDE_DIR}"
        "${DSSIM_DAWN_SRC_INCLUDE_DIR}"
    )

    target_link_libraries(dssim_gpu_dawn_checksum PRIVATE
        "${DSSIM_DAWN_WEBGPU_DAWN_LIB}"
        "${DSSIM_DAWN_DAWN_PROC_LIB}"
        "${DSSIM_DAWN_DAWN_NATIVE_LIB}"
    )

    if(MSVC)
        target_compile_options(dssim_gpu_dawn_checksum PRIVATE /W4 /EHsc)
    else()
        target_compile_options(dssim_gpu_dawn_checksum PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(WIN32)
        target_link_libraries(dssim_gpu_dawn_checksum PRIVATE dxguid)
    endif()

    add_custom_command(TARGET dssim_gpu_dawn_checksum POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:dssim_gpu_dawn_checksum>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DSSIM_GPU_STAGE0_SHADER}"
            "$<TARGET_FILE_DIR:dssim_gpu_dawn_checksum>/shaders/stage0_absdiff.wgsl"
    )
endif()
