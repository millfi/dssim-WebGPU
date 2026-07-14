find_package(Vulkan 1.3 REQUIRED COMPONENTS glslc)

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
if(NOT DSSIM_FFMPEG_INCLUDE_DIR OR NOT DSSIM_FFMPEG_AVCODEC_LIBRARY OR
   NOT DSSIM_FFMPEG_AVFORMAT_LIBRARY OR NOT DSSIM_FFMPEG_AVUTIL_LIBRARY)
    message(FATAL_ERROR
        "Vulkan Video FFmpeg development files were not found under ${DSSIM_FFMPEG_ROOT}. "
        "Run tools/build_ffmpeg_minimal.ps1 -Linkage Dynamic first.")
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

add_executable(dssim_webgpu
    cli_options.cpp
    dssim-WebGPU.cpp
    png_loader.cpp
    video_decoder.cpp
)
set_target_properties(dssim_webgpu PROPERTIES
    OUTPUT_NAME "dssim-WebGPU"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src_gpu/$<CONFIG>"
)

target_compile_features(dssim_webgpu PRIVATE cxx_std_20)
target_include_directories(dssim_webgpu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${DSSIM_FFMPEG_INCLUDE_DIR}"
)
target_link_libraries(dssim_webgpu PRIVATE
    Vulkan::Vulkan
    "${DSSIM_PNG_TARGET}"
    "${DSSIM_FFMPEG_AVCODEC_LIBRARY}"
    "${DSSIM_FFMPEG_AVFORMAT_LIBRARY}"
    "${DSSIM_FFMPEG_AVUTIL_LIBRARY}"
)

if(WIN32)
    add_custom_command(TARGET dssim_webgpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${DSSIM_FFMPEG_ROOT}/bin"
            "$<TARGET_FILE_DIR:dssim_webgpu>"
        COMMENT "Copying dynamically linked FFmpeg Vulkan Video DLLs")
endif()

if(MSVC)
    target_compile_options(dssim_webgpu PRIVATE /W4 /EHsc)
else()
    target_compile_options(dssim_webgpu PRIVATE -Wall -Wextra -Wpedantic)
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

add_custom_target(dssim_webgpu_shaders
    DEPENDS ${DSSIM_GPU_SHADER_OUTPUTS}
)
add_dependencies(dssim_webgpu dssim_webgpu_shaders)
