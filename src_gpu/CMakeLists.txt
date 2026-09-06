find_package(Vulkan 1.4.357 REQUIRED)
find_program(DSSIM_SLANGC NAMES slangc
    HINTS "${CMAKE_SOURCE_DIR}/third_party/slang-2026.14.1/bin" "$ENV{VULKAN_SDK}/Bin" REQUIRED)
execute_process(COMMAND "${DSSIM_SLANGC}" -version OUTPUT_VARIABLE slang_version
    ERROR_VARIABLE slang_version_error RESULT_VARIABLE slang_version_result)
string(APPEND slang_version "${slang_version_error}")
if(NOT slang_version_result EQUAL 0 OR NOT slang_version MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)")
    message(FATAL_ERROR "Cannot determine Slang version: ${slang_version}")
endif()
if(CMAKE_MATCH_1 VERSION_LESS "2026.14.1")
    message(FATAL_ERROR "Slang 2026.14.1+ is required. Run tools/setup_slang.ps1 or set DSSIM_SLANGC.")
endif()
find_program(DSSIM_SPIRV_VAL NAMES spirv-val HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)
add_library(NoGraphicsAPI STATIC
    "${CMAKE_SOURCE_DIR}/third_party/NoGraphicsAPI/src/NoGraphicsAPI.cpp")
target_include_directories(NoGraphicsAPI PUBLIC
    "${CMAKE_SOURCE_DIR}/third_party/NoGraphicsAPI/include")
target_link_libraries(NoGraphicsAPI PUBLIC Vulkan::Vulkan)
target_compile_features(NoGraphicsAPI PUBLIC cxx_std_20)


set(DSSIM_FFMPEG_ROOT "${CMAKE_SOURCE_DIR}/third_party/ffmpeg-8.1.2-shared" CACHE PATH
    "FFmpeg prefix containing the Vulkan Video development headers and import libraries")
find_path(DSSIM_FFMPEG_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS "${DSSIM_FFMPEG_ROOT}/include"
    NO_DEFAULT_PATH)
find_library(DSSIM_FFMPEG_AVCODEC_LIBRARY NAMES avcodec avcodec-62
    PATHS "${DSSIM_FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
find_library(DSSIM_FFMPEG_AVFORMAT_LIBRARY NAMES avformat avformat-62
    PATHS "${DSSIM_FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
find_library(DSSIM_FFMPEG_AVUTIL_LIBRARY NAMES avutil avutil-60
    PATHS "${DSSIM_FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
find_library(DSSIM_FFMPEG_SWSCALE_LIBRARY NAMES swscale swscale-9
    PATHS "${DSSIM_FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT DSSIM_FFMPEG_INCLUDE_DIR OR NOT DSSIM_FFMPEG_AVCODEC_LIBRARY OR
   NOT DSSIM_FFMPEG_AVFORMAT_LIBRARY OR NOT DSSIM_FFMPEG_AVUTIL_LIBRARY OR
   NOT DSSIM_FFMPEG_SWSCALE_LIBRARY)
    message(FATAL_ERROR
        "Vulkan Video FFmpeg development files were not found under ${DSSIM_FFMPEG_ROOT}. "
        "Run tools/build_ffmpeg_minimal.ps1 -Linkage Dynamic first.")
endif()

add_executable(dssim_vulkan
    cli_options.cpp
    dssim-Vulkan.cpp
    image_loader.cpp
    video_decoder.cpp
)
set_target_properties(dssim_vulkan PROPERTIES
    OUTPUT_NAME "dssim-Vulkan"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src_gpu/$<CONFIG>"
)

target_compile_features(dssim_vulkan PRIVATE cxx_std_20)
target_include_directories(dssim_vulkan PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${DSSIM_FFMPEG_INCLUDE_DIR}"
)
target_link_libraries(dssim_vulkan PRIVATE
    NoGraphicsAPI
    "${DSSIM_FFMPEG_AVCODEC_LIBRARY}"
    "${DSSIM_FFMPEG_AVFORMAT_LIBRARY}"
    "${DSSIM_FFMPEG_AVUTIL_LIBRARY}"
    "${DSSIM_FFMPEG_SWSCALE_LIBRARY}"
)

if(WIN32)
    add_custom_command(TARGET dssim_vulkan POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${DSSIM_FFMPEG_ROOT}/bin"
            "$<TARGET_FILE_DIR:dssim_vulkan>"
        COMMENT "Copying dynamically linked FFmpeg Vulkan Video DLLs")
endif()

if(MSVC)
    target_compile_options(dssim_vulkan PRIVATE /W4 /EHsc)
else()
    target_compile_options(dssim_vulkan PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(DSSIM_GPU_SHADER_NAMES
    rgba8_to_linear
    vulkan_yuv_to_rgba8
    downsample_2x2
    lab_preprocess
    stage0_absdiff
    stage0_score
    reduce_sum
    reduce_abs_deviation
)
set(DSSIM_GPU_SHADER_OUTPUT_DIR
    "${CMAKE_BINARY_DIR}/src_gpu/$<CONFIG>/shaders"
)
set(DSSIM_GPU_SHADER_OUTPUTS)

foreach(DSSIM_GPU_SHADER_NAME IN LISTS DSSIM_GPU_SHADER_NAMES)
    set(DSSIM_GPU_SHADER_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${DSSIM_GPU_SHADER_NAME}.comp"
    )
    set(DSSIM_GPU_SHADER_OUTPUT
        "${DSSIM_GPU_SHADER_OUTPUT_DIR}/${DSSIM_GPU_SHADER_NAME}.spv"
    )
    add_custom_command(
        OUTPUT "${DSSIM_GPU_SHADER_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${DSSIM_GPU_SHADER_OUTPUT_DIR}"
        COMMAND "${DSSIM_SLANGC}"
            -lang slang -target spirv -profile spirv_1_5
            -emit-spirv-directly -fvk-use-entrypoint-name -fvk-use-c-layout
            -matrix-layout-row-major -capability spvDescriptorHeapEXT
            -entry computeMain -stage compute -O2
            "${DSSIM_GPU_SHADER_SOURCE}"
            -o "${DSSIM_GPU_SHADER_OUTPUT}"
        COMMAND "${DSSIM_SPIRV_VAL}" --target-env vulkan1.4 --scalar-block-layout
            "${DSSIM_GPU_SHADER_OUTPUT}"
        DEPENDS "${DSSIM_GPU_SHADER_SOURCE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_root.h"
        COMMENT "Compiling ${DSSIM_GPU_SHADER_NAME}.comp to SPIR-V"
        VERBATIM
    )
    list(APPEND DSSIM_GPU_SHADER_OUTPUTS "${DSSIM_GPU_SHADER_OUTPUT}")
endforeach()

add_custom_target(dssim_vulkan_shaders
    DEPENDS ${DSSIM_GPU_SHADER_OUTPUTS}
)
add_dependencies(dssim_vulkan dssim_vulkan_shaders)
