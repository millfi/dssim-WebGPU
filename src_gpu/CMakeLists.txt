cmake_minimum_required(VERSION 3.24)


option(DSSIM_ENABLE_DAWN_SAMPLE "Build Dawn native compute sample target" ON)
option(DSSIM_AUTO_INSTALL_DAWN "Automatically fetch and build Dawn when missing (Windows only)" ON)
set(DSSIM_DAWN_ROOT "${CMAKE_SOURCE_DIR}/third_party/dawn" CACHE PATH "Path to Dawn source root")
set(DSSIM_DAWN_OUT_DIR "${DSSIM_DAWN_ROOT}/out/Release" CACHE PATH "Path to Dawn GN build output directory")
set(DSSIM_BUILD_DAWN_SAMPLE OFF)

function(dssim_probe_dawn)
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
        DSSIM_DAWN_INCLUDE_DIR AND
        DSSIM_DAWN_SRC_INCLUDE_DIR AND
        DSSIM_DAWN_WEBGPU_DAWN_LIB AND
        DSSIM_DAWN_DAWN_PROC_LIB AND
        DSSIM_DAWN_DAWN_NATIVE_LIB
    )
        set(DSSIM_BUILD_DAWN_SAMPLE ON PARENT_SCOPE)
    else()
        set(DSSIM_BUILD_DAWN_SAMPLE OFF PARENT_SCOPE)
    endif()

    set(DSSIM_DAWN_INCLUDE_DIR "${DSSIM_DAWN_INCLUDE_DIR}" PARENT_SCOPE)
    set(DSSIM_DAWN_SRC_INCLUDE_DIR "${DSSIM_DAWN_SRC_INCLUDE_DIR}" PARENT_SCOPE)
    set(DSSIM_DAWN_WEBGPU_DAWN_LIB "${DSSIM_DAWN_WEBGPU_DAWN_LIB}" PARENT_SCOPE)
    set(DSSIM_DAWN_DAWN_PROC_LIB "${DSSIM_DAWN_DAWN_PROC_LIB}" PARENT_SCOPE)
    set(DSSIM_DAWN_DAWN_NATIVE_LIB "${DSSIM_DAWN_DAWN_NATIVE_LIB}" PARENT_SCOPE)
endfunction()

if(DSSIM_ENABLE_DAWN_SAMPLE)
    include(FetchContent)

    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(dssim_zlib
        URL https://github.com/madler/zlib/archive/refs/tags/v1.3.1.zip
    )
    FetchContent_MakeAvailable(dssim_zlib)
    if(TARGET zlibstatic AND NOT TARGET ZLIB::ZLIB)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
    endif()

    set(PNG_BUILD_ZLIB ON CACHE BOOL "" FORCE)
    set(PNG_SHARED OFF CACHE BOOL "" FORCE)
    set(PNG_STATIC ON CACHE BOOL "" FORCE)
    set(PNG_TESTS OFF CACHE BOOL "" FORCE)
    set(PNG_TOOLS OFF CACHE BOOL "" FORCE)
    set(PNG_FRAMEWORK OFF CACHE BOOL "" FORCE)
    set(PNG_HARDWARE_OPTIMIZATIONS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(dssim_libpng
        URL https://github.com/pnggroup/libpng/archive/refs/tags/v1.6.44.zip
    )
    FetchContent_MakeAvailable(dssim_libpng)

    if(TARGET PNG::PNG)
        set(DSSIM_PNG_TARGET PNG::PNG)
    elseif(TARGET png_static)
        set(DSSIM_PNG_TARGET png_static)
    elseif(TARGET png)
        set(DSSIM_PNG_TARGET png)
    else()
        message(FATAL_ERROR "libpng target not found after FetchContent")
    endif()

    dssim_probe_dawn()

    if(NOT DSSIM_BUILD_DAWN_SAMPLE AND DSSIM_AUTO_INSTALL_DAWN)
        if(NOT WIN32)
            message(FATAL_ERROR "DSSIM_AUTO_INSTALL_DAWN=ON is currently supported on Windows only.")
        endif()

        find_program(DSSIM_PWSH NAMES pwsh powershell REQUIRED)
        message(STATUS "Dawn not found. Running automatic Dawn setup script.")
        execute_process(
            COMMAND
                "${DSSIM_PWSH}" -NoProfile -ExecutionPolicy Bypass
                -File "${CMAKE_SOURCE_DIR}/tools/install_dawn.ps1"
                -DawnRoot "${DSSIM_DAWN_ROOT}"
                -DawnOutDir "${DSSIM_DAWN_OUT_DIR}"
                -DepotToolsDir "${CMAKE_SOURCE_DIR}/third_party/depot_tools"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RESULT_VARIABLE DSSIM_DAWN_SETUP_RESULT
        )
        if(NOT DSSIM_DAWN_SETUP_RESULT EQUAL 0)
            message(FATAL_ERROR
                "Automatic Dawn setup failed (exit code ${DSSIM_DAWN_SETUP_RESULT}). "
                "Run tools/install_dawn.ps1 manually or set DSSIM_DAWN_ROOT/DSSIM_DAWN_OUT_DIR."
            )
        endif()
        dssim_probe_dawn()
    endif()

    if(NOT DSSIM_BUILD_DAWN_SAMPLE)
        message(WARNING
            "Dawn sample requested but required headers/libs were not found. "
            "Skipping dssim_gpu_dawn_checksum. "
            "Set DSSIM_DAWN_ROOT and DSSIM_DAWN_OUT_DIR to a valid Dawn build, "
            "or enable -DDSSIM_AUTO_INSTALL_DAWN=ON on Windows."
        )
    endif()
endif()

if(DSSIM_BUILD_DAWN_SAMPLE)
    add_executable(dssim_gpu_dawn_checksum
        dawn_checksum.cpp
        png_loader.cpp
    )
    set(DSSIM_GPU_STAGE0_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/stage0_absdiff.wgsl")
    set(DSSIM_GPU_DOWNSAMPLE_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/downsample_2x2.wgsl")
    set(DSSIM_GPU_LAB_PREPROCESS_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lab_preprocess.wgsl")

    target_compile_features(dssim_gpu_dawn_checksum PRIVATE cxx_std_20)
    target_include_directories(dssim_gpu_dawn_checksum PRIVATE
        "${DSSIM_DAWN_INCLUDE_DIR}"
        "${DSSIM_DAWN_SRC_INCLUDE_DIR}"
    )

    target_link_libraries(dssim_gpu_dawn_checksum PRIVATE
        "${DSSIM_DAWN_WEBGPU_DAWN_LIB}"
        "${DSSIM_DAWN_DAWN_PROC_LIB}"
        "${DSSIM_DAWN_DAWN_NATIVE_LIB}"
        "${DSSIM_PNG_TARGET}"
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
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DSSIM_GPU_DOWNSAMPLE_SHADER}"
            "$<TARGET_FILE_DIR:dssim_gpu_dawn_checksum>/shaders/downsample_2x2.wgsl"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DSSIM_GPU_LAB_PREPROCESS_SHADER}"
            "$<TARGET_FILE_DIR:dssim_gpu_dawn_checksum>/shaders/lab_preprocess.wgsl"
    )
endif()
