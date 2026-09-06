find_package(Vulkan 1.3 REQUIRED COMPONENTS glslc)

set(DSSIM_FFMPEG_ROOT "${CMAKE_SOURCE_DIR}/third_party/ffmpeg-gpu-shared")
set(_ffmpeg_marker "${DSSIM_FFMPEG_ROOT}/dssim-ffmpeg-variant.txt")
if(NOT EXISTS "${_ffmpeg_marker}")
    message(FATAL_ERROR "Build patched GPU FFmpeg DLLs with tools/build_ffmpeg_minimal.ps1 -Variant Gpu first.")
endif()
file(READ "${_ffmpeg_marker}" _ffmpeg_variant)
string(STRIP "${_ffmpeg_variant}" _ffmpeg_variant)
if(NOT _ffmpeg_variant STREQUAL "gpu-shared-v1")
    message(FATAL_ERROR "GPU FFmpeg variant mismatch")
endif()
# Invalidate entries that could still select the legacy shared build.
foreach(_component INCLUDE_DIR AVCODEC_LIBRARY AVFORMAT_LIBRARY AVUTIL_LIBRARY SWSCALE_LIBRARY)
    unset(DSSIM_FFMPEG_${_component} CACHE)
endforeach()
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
        "Run tools/build_ffmpeg_minimal.ps1 -Variant Gpu first.")
endif()

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${Vulkan_INCLUDE_DIRS}")
check_cxx_source_compiles([=[
    #include <vulkan/vulkan.h>
    int main() {
        VkShaderEXT shader = VK_NULL_HANDLE;
        VkPhysicalDeviceShaderObjectFeaturesEXT features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
        return shader == VK_NULL_HANDLE && features.sType != 0 ? 0 : 1;
    }
]=] DSSIM_HAS_SHADER_OBJECT_HEADERS)
unset(CMAKE_REQUIRED_INCLUDES)
if(NOT DSSIM_HAS_SHADER_OBJECT_HEADERS)
    message(FATAL_ERROR
        "The Vulkan SDK headers do not provide VK_EXT_shader_object. "
        "Install a current Vulkan SDK."
    )
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
    Vulkan::Vulkan
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
        COMMAND "$<TARGET_FILE:Vulkan::glslc>"
            --target-env=vulkan1.2
            -fshader-stage=compute
            -O
            "${DSSIM_GPU_SHADER_SOURCE}"
            -o "${DSSIM_GPU_SHADER_OUTPUT}"
        DEPENDS "${DSSIM_GPU_SHADER_SOURCE}"
        COMMENT "Compiling ${DSSIM_GPU_SHADER_NAME}.comp to SPIR-V"
        VERBATIM
    )
    list(APPEND DSSIM_GPU_SHADER_OUTPUTS "${DSSIM_GPU_SHADER_OUTPUT}")
endforeach()

add_custom_target(dssim_vulkan_shaders
    DEPENDS ${DSSIM_GPU_SHADER_OUTPUTS}
)
add_dependencies(dssim_vulkan dssim_vulkan_shaders)
