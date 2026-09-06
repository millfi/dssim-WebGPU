#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <numeric>
#include <span>

#include <NoGraphicsAPI/NoGraphicsAPI_vulkan.hpp>
#include "shaders/compute_root.h"

#include "cli_options.h"
#include "image_loader.h"
#include "video_decoder.h"
using namespace std::chrono;
namespace {

constexpr std::uint32_t kStage0QScale = 100000000u;
constexpr std::uint32_t kStage0WindowRadius = 2u;
constexpr std::uint32_t kStage0WindowSize = kStage0WindowRadius * 2u + 1u;
constexpr std::array<double, 5> kDefaultScaleWeights = {0.028, 0.197, 0.322, 0.298, 0.155};

struct LinearRgba {
    LinearRgba() noexcept {}

    float r;
    float g;
    float b;
    float a;
};

struct ScaleOutputs {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::size_t elemCount = 0;
    std::vector<float> ssimMap;
    std::vector<float> mu1;
    std::vector<float> mu2;
    std::vector<float> var1;
    std::vector<float> var2;
    std::vector<float> cov12;
    double meanSsim = 0.0;
    double ssimScore = 0.0;
    // profiling
    std::chrono::milliseconds createShaderModule_time{0};
    std::chrono::milliseconds createPSO_time{0};
    std::chrono::milliseconds createBuffers_time{0};
    std::chrono::milliseconds writeInputBuffers_time{0};
    std::chrono::milliseconds createPipelineLayouts_time{0};
    std::chrono::milliseconds createBindGroups_time{0};
    std::chrono::milliseconds dispatchAndSubmit_time{0};
    std::chrono::milliseconds readback_time{0};
    std::chrono::milliseconds postProcess_time{0};
    double gpuTimestampMs = 0.0;
    double postProcessBaseScaleMs = 0.0;
    double postProcessRemainingScalesMs = 0.0;
};

struct MultiScaleOutputs {
    std::vector<ScaleOutputs> scales;
    double weightedSsim = 0.0;
    double score = 0.0;
};

struct DebugDumpInfo {
    std::filesystem::path stage0DssimPath;
    std::filesystem::path stage0Mu1Path;
    std::filesystem::path stage0Mu2Path;
    std::filesystem::path stage0Var1Path;
    std::filesystem::path stage0Var2Path;
    std::filesystem::path stage0Cov12Path;
    std::filesystem::path stage1DssimPath;
    std::filesystem::path image1Scale1Path;
    std::filesystem::path image2Scale1Path;
    std::filesystem::path image1RgbaPath;
    std::filesystem::path image2RgbaPath;
    std::size_t stage0ElemCount = 0;
    std::size_t stage1ElemCount = 0;
};

struct DecodedInputInfo {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    std::size_t byteCount = 0;
};

struct DownsampleOutputs {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<LinearRgba> pixels;
    // profiling
    std::chrono::milliseconds createShaderModule_time{0};
    std::chrono::milliseconds createPSO_time{0};
    std::chrono::milliseconds createBuffers_time{0};
    std::chrono::milliseconds writeInputBuffers_time{0};
    std::chrono::milliseconds createPipelineLayouts_time{0};
    std::chrono::milliseconds createBindGroups_time{0};
    std::chrono::milliseconds dispatchAndSubmit_time{0};
    std::chrono::milliseconds readback_time{0};
    std::chrono::milliseconds postProcess_time{0};
};

struct ProfilingSummary {
    long long decodeDoneToScoreMs = 0;
    std::chrono::milliseconds createShaderModuleTime{0};
    std::chrono::milliseconds createPSOTime{0};
    std::chrono::milliseconds createBuffersTime{0};
    std::chrono::milliseconds writeInputBuffersTime{0};
    std::chrono::milliseconds createPipelineLayoutsTime{0};
    std::chrono::milliseconds createBindGroupsTime{0};
    std::chrono::milliseconds dispatchAndSubmitTime{0};
    std::chrono::milliseconds readbackTime{0};
    std::chrono::milliseconds postProcessTime{0};
    std::chrono::milliseconds otherTime{0};
    double gpuTimestampMs = 0.0;
    double postProcessBaseScaleMs = 0.0;
    double postProcessRemainingScalesMs = 0.0;
};

struct RgbaPairComparisonResult {
    MultiScaleOutputs compute;
    ProfilingSummary profiling;
    std::vector<LinearRgba> debugScale1Image1;
    std::vector<LinearRgba> debugScale1Image2;
};

struct GpuBuffer {
    gpu::GpuHeap heap{};
    std::uint64_t size = 0;
    void* mapped = nullptr;

    GpuBuffer() = default;
    ~GpuBuffer() { Reset(); }
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&& other) noexcept { *this = std::move(other); }
    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            Reset();
            heap = std::exchange(other.heap, {});
            size = std::exchange(other.size, 0);
            mapped = std::exchange(other.mapped, nullptr);
        }
        return *this;
    }
    explicit operator bool() const noexcept { return heap.owner != nullptr; }
    void Reset() noexcept {
        if (heap.owner) gpu::destroy_gpu_heap(heap);
        heap = {};
        size = 0;
        mapped = nullptr;
    }
};

struct ComputeArenas {
    std::uint64_t workspaceCapacity = 0;
    std::uint64_t uploadCapacity = 0;
    std::uint64_t readbackCapacity = 0;
    GpuBuffer workspace;
    GpuBuffer upload;
    GpuBuffer readback;
};

struct GpuSession {
    gpu::Device* gpuDevice = nullptr;
    gpu::CommandBuffer* commands = nullptr;
    gpu::TimelineSemaphore* completion = nullptr;
    std::uint64_t completionValue = 0;
    ComputeRoot root{};
    gpu::GpuHeap videoTextureHeap{};
    gpu::GpuHeap videoSamplerHeap{};
    PFN_vkWriteResourceDescriptorsEXT writeResourceDescriptors = nullptr;
    // Borrowed handles for FFmpeg interoperability and timestamp queries.
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t queueFamilyIndex = 0;
    VkQueueFlags computeQueueFlags = 0;
    std::uint32_t videoDecodeQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    VkQueueFlags videoDecodeQueueFlags = 0;
    VkVideoCodecOperationFlagsKHR videoDecodeCaps = 0;
    std::uint32_t videoDecodeQueueFamilyIndexSecondary = VK_QUEUE_FAMILY_IGNORED;
    VkQueueFlags videoDecodeQueueFlagsSecondary = 0;
    VkVideoCodecOperationFlagsKHR videoDecodeCapsSecondary = 0;
    bool videoSupported = false;
    std::vector<const char*> videoDeviceExtensions;
    VkPhysicalDeviceProperties physicalDeviceProperties{};
    std::string adapterName = "unknown";
    bool timestampQueryEnabled = false;
    std::uint32_t timestampValidBits = 0;
    VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;


    gpu::PSO* preprocessShader = nullptr;
    gpu::PSO* stage0Shader = nullptr;
    gpu::PSO* stage0ScoreShader = nullptr;
    gpu::PSO* reduceSumShader = nullptr;
    gpu::PSO* reduceAbsDeviationShader = nullptr;
    gpu::PSO* downsampleShader = nullptr;
    gpu::PSO* rgba8ToLinearShader = nullptr;
    gpu::PSO* vulkanYuvToRgbaShader = nullptr;
    GpuBuffer srgbToLinearLutBuffer;

    std::unique_ptr<ComputeArenas> debugComputeArenas;
    std::unique_ptr<ComputeArenas> batchComputeArenas;

    ProfilingSummary initProfiling;

    GpuSession() = default;
    ~GpuSession();
    GpuSession(const GpuSession&) = delete;
    GpuSession& operator=(const GpuSession&) = delete;
};

struct ProfilingBuckets {
    double totalMs = 0.0;
    double pipelineSetupMs = 0.0;
    double resourcePrepMs = 0.0;
    double gpuSubmitWaitMs = 0.0;
    double gpuTimestampMs = 0.0;
    double cpuPostProcessMs = 0.0;
    double otherMs = 0.0;
};

std::string EscapeJson(const std::string& input) {
    std::ostringstream os;
    for (unsigned char c : input) {
        switch (c) {
            case '"':
                os << "\\\"";
                break;
            case '\\':
                os << "\\\\";
                break;
            case '\b':
                os << "\\b";
                break;
            case '\f':
                os << "\\f";
                break;
            case '\n':
                os << "\\n";
                break;
            case '\r':
                os << "\\r";
                break;
            case '\t':
                os << "\\t";
                break;
            default:
                if (c < 0x20) {
                    os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(c) << std::dec;
                } else {
                    os << static_cast<char>(c);
                }
                break;
        }
    }
    return os.str();
}

std::string ToHexU64(double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "double/u64 size mismatch");
    std::memcpy(&bits, &value, sizeof(bits));

    std::ostringstream os;
    os << "0x" << std::uppercase << std::hex << std::setw(16) << std::setfill('0') << bits;
    return os.str();
}

std::filesystem::path ResolveShaderPath(
    const std::filesystem::path& executableArg,
    const std::string& shaderFileName) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    const std::filesystem::path exeAbs = std::filesystem::absolute(executableArg);
    const std::filesystem::path exeDir = exeAbs.parent_path();

    const std::array<std::filesystem::path, 4> candidates = {
        exeDir / "shaders" / shaderFileName,
        exeDir / shaderFileName,
        cwd / "src_gpu" / "shaders" / shaderFileName,
        cwd / "build" / "src_gpu" / "shaders" / shaderFileName,
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    std::ostringstream message;
    message << "shader file not found: " << shaderFileName << ". searched:";
    for (const auto& candidate : candidates) {
        message << " " << candidate.string();
    }
    throw std::runtime_error(message.str());
}

float SrgbToLinear(float c) {
    if (c <= 0.04045f) {
        return c / 12.92f;
    }
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

float LinearToSrgb(float c) {
    if (c <= 0.0031308f) {
        return c * 12.92f;
    }
    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

std::uint8_t ToUnorm8(float value) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::lround(clamped * 255.0f));
}

template <typename Function>
void ParallelFor(
    std::size_t itemCount,
    Function&& function) {
    const unsigned int hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t maxWorkers = std::max<std::size_t>(1u, hardwareThreads / 2u);
    const std::size_t workerCount = std::min(maxWorkers, itemCount);
    if (workerCount == 0u) {
        return;
    }
    if (workerCount == 1u) {
        function(0u, itemCount);
        return;
    }

    const std::size_t itemsPerWorker = (itemCount + workerCount - 1u) / workerCount;
    std::vector<std::thread> workers;
    workers.reserve(workerCount - 1u);
    for (std::size_t worker = 1u; worker < workerCount; ++worker) {
        const std::size_t begin = worker * itemsPerWorker;
        const std::size_t end = std::min(itemCount, begin + itemsPerWorker);
        if (begin < end) {
            workers.emplace_back(function, begin, end);
        }
    }

    function(0u, std::min(itemCount, itemsPerWorker));
    for (auto& worker : workers) {
        worker.join();
    }
}

const std::array<float, 256>& SrgbToLinearLut() {
    static const std::array<float, 256> lut = [] {
        std::array<float, 256> table{};
        for (std::size_t i = 0; i < table.size(); ++i) {
            table[i] = SrgbToLinear(static_cast<float>(i) / 255.0f);
        }
        return table;
    }();
    return lut;
}

std::vector<LinearRgba> ConvertRgba8ToLinearPlu(
    std::span<const std::uint8_t> bytes) {
    if ((bytes.size() % 4) != 0) {
        throw std::runtime_error("rgba8 byte count is not divisible by 4");
    }

    const std::size_t pixelCount = bytes.size() / 4;
    std::vector<LinearRgba> out(pixelCount);
    const auto& srgbToLinearLut = SrgbToLinearLut();
    ParallelFor(pixelCount, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const std::size_t base = i * 4;
            const float a = static_cast<float>(bytes[base + 3]) / 255.0f;
            out[i].r = srgbToLinearLut[bytes[base + 0]] * a;
            out[i].g = srgbToLinearLut[bytes[base + 1]] * a;
            out[i].b = srgbToLinearLut[bytes[base + 2]] * a;
            out[i].a = a;
        }
    });
    return out;
}

std::vector<std::uint8_t> ConvertLinearPluToRgba8(const std::vector<LinearRgba>& pixels) {
    std::vector<std::uint8_t> out(pixels.size() * 4);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        const float a = std::clamp(pixels[i].a, 0.0f, 1.0f);
        const float invA = (a > 1.0e-8f) ? (1.0f / a) : 0.0f;
        const float r = std::clamp(pixels[i].r * invA, 0.0f, 1.0f);
        const float g = std::clamp(pixels[i].g * invA, 0.0f, 1.0f);
        const float b = std::clamp(pixels[i].b * invA, 0.0f, 1.0f);
        out[i * 4 + 0] = ToUnorm8(LinearToSrgb(r));
        out[i * 4 + 1] = ToUnorm8(LinearToSrgb(g));
        out[i * 4 + 2] = ToUnorm8(LinearToSrgb(b));
        out[i * 4 + 3] = ToUnorm8(a);
    }
    return out;
}

void WriteF32LeBuffer(const std::filesystem::path& outPath, const std::vector<float>& values) {
    const auto parent = outPath.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open output: " + outPath.string());
    }

    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    if (!out) {
        throw std::runtime_error("failed to write output: " + outPath.string());
    }
}

void WriteU8Buffer(
    const std::filesystem::path& outPath,
    std::span<const std::uint8_t> values) {
    const auto parent = outPath.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open output: " + outPath.string());
    }

    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()));
    }

    if (!out) {
        throw std::runtime_error("failed to write output: " + outPath.string());
    }
}

std::string BuildJson(
    const CliOptions& options,
    const std::string& adapterName,
    const DecodedInputInfo& decoded1,
    const DecodedInputInfo& decoded2,
    const MultiScaleOutputs& compute,
    const ProfilingSummary& profiling,
    const DebugDumpInfo* debugInfo) {
    const auto abs1 = std::filesystem::absolute(options.image1).string();
    const auto abs2 = std::filesystem::absolute(options.image2).string();
    std::string absOut;
    if (!options.out.empty()) {
        absOut = std::filesystem::absolute(options.out).string();
    }

    std::ostringstream command;
    command << "dssim-Vulkan \"" << abs1 << "\" \"" << abs2 << "\"";
    if (!absOut.empty()) {
        command << " --out \"" << absOut << "\"";
    }
    if (options.debugDumpEnabled) {
        const auto absDebug = std::filesystem::absolute(options.debugDumpDir).string();
        command << " --debug-dump-dir \"" << absDebug << "\"";
    }
    if (options.profilingEnabled) {
        command << " --profiling";
    }

    std::ostringstream os;
    os << "{\n";
    os << "  \"schema_version\": 1,\n";
    os << "  \"engine\": \"gpu-vulkan-spirv-dssim-ms-stage5x5-gaussian-linear\",\n";
    os << "  \"status\": \"ok\",\n";
    os << "  \"input\": {\n";
    os << "    \"image1\": \"" << EscapeJson(abs1) << "\",\n";
    os << "    \"image2\": \"" << EscapeJson(abs2) << "\"\n";
    os << "  },\n";
    os << "  \"decoded_input\": {\n";
    os << "    \"image1\": {\n";
    os << "      \"width\": " << decoded1.width << ",\n";
    os << "      \"height\": " << decoded1.height << ",\n";
    os << "      \"channels\": " << decoded1.channels << ",\n";
    os << "      \"bytes\": " << decoded1.byteCount << "\n";
    os << "    },\n";
    os << "    \"image2\": {\n";
    os << "      \"width\": " << decoded2.width << ",\n";
    os << "      \"height\": " << decoded2.height << ",\n";
    os << "      \"channels\": " << decoded2.channels << ",\n";
    os << "      \"bytes\": " << decoded2.byteCount << "\n";
    os << "    }\n";
    os << "  },\n";
    os << "  \"command\": \"" << EscapeJson(command.str()) << "\",\n";
    os << "  \"version\": \"vulkan-shader-object-dssim-ms-stage5x5-gaussian-linear-1\",\n";
    os << "  \"result\": {\n";
    std::ostringstream scoreText;
    scoreText << std::fixed << std::setprecision(8) << compute.score;
    os << "    \"score_source\": \"gpu-reference-like-ms-ssim-provisional\",\n";
    os << "    \"score_text\": \"" << scoreText.str() << "\",\n";
    os << "    \"score_f64\": " << std::setprecision(17) << compute.score << ",\n";
    os << "    \"score_bits_u64\": \"" << ToHexU64(compute.score) << "\",\n";
    os << "    \"compared_path\": \"" << EscapeJson(abs2) << "\",\n";
    os << "    \"gpu_scales\": [\n";
    for (std::size_t i = 0; i < compute.scales.size(); ++i) {
        const auto& scale = compute.scales[i];
        os << "      {\n";
        os << "        \"level\": " << i << ",\n";
        os << "        \"width\": " << scale.width << ",\n";
        os << "        \"height\": " << scale.height << ",\n";
        os << "        \"metric\": \"dssim_5x5_gaussian_luma_linear_srgb\",\n";
        os << "        \"window_radius\": " << kStage0WindowRadius << ",\n";
        os << "        \"window_size\": " << kStage0WindowSize << ",\n";
        os << "        \"window_type\": \"gaussian_blur_kernel_x2\",\n";
        os << "        \"weight\": " << std::setprecision(17) << kDefaultScaleWeights[i] << ",\n";
        os << "        \"elem_count\": " << scale.elemCount << ",\n";
        os << "        \"mean_ssim_f64\": " << std::setprecision(17) << scale.meanSsim << ",\n";
        os << "        \"ssim_score_f64\": " << std::setprecision(17) << scale.ssimScore << "\n";
        os << "      }";
        if (i + 1 < compute.scales.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "    ],\n";
    os << "    \"aggregation\": {\n";
    os << "      \"method\": \"reference_like_weighted_ssim_to_dssim\",\n";
    os << "      \"used_scale_count\": " << compute.scales.size() << ",\n";
    os << "      \"weighted_ssim_f64\": " << std::setprecision(17) << compute.weightedSsim << "\n";
    os << "    }\n";
    os << "  },\n";
    os << "  \"adapter\": \"" << EscapeJson(adapterName) << "\",\n";
    os << "  \"profiling\": {\n";
    os << "    \"decode_done_to_score_ms\": " << profiling.decodeDoneToScoreMs << ",\n";
    os << "    \"create_shader_module_ms\": " << profiling.createShaderModuleTime.count() << ",\n";
    os << "    \"create_pso_ms\": " << profiling.createPSOTime.count() << ",\n";
    os << "    \"create_buffer_ms\": " << profiling.createBuffersTime.count() << ",\n";
    os << "    \"write_input_buffer_ms\": " << profiling.writeInputBuffersTime.count() << ",\n";
    os << "    \"create_pipeline_layout_ms\": " << profiling.createPipelineLayoutsTime.count() << ",\n";
    os << "    \"create_bind_group_ms\": " << profiling.createBindGroupsTime.count() << ",\n";
    os << "    \"dispatch_and_submit_ms\": " << profiling.dispatchAndSubmitTime.count() << ",\n";
    os << "    \"readback_ms\": " << profiling.readbackTime.count() << ",\n";
    os << "    \"gpu_submit_wait_ms\": "
       << (profiling.dispatchAndSubmitTime + profiling.readbackTime).count() << ",\n";
    os << "    \"gpu_timestamp_ms\": " << std::setprecision(9)
       << profiling.gpuTimestampMs << ",\n";
    os << "    \"post_process_base_scale_ms\": " << std::setprecision(9)
       << profiling.postProcessBaseScaleMs << ",\n";
    os << "    \"post_process_remaining_scales_ms\": " << std::setprecision(9)
       << profiling.postProcessRemainingScalesMs << ",\n";
    os << "    \"post_process_ms\": " << profiling.postProcessTime.count() << ",\n";
    os << "    \"other_ms\": " << profiling.otherTime.count() << "\n";
    os << "  }";

    if (debugInfo != nullptr) {
        os << ",\n";
        os << "  \"debug_dumps\": {\n";
        os << "    \"image1_rgba8\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->image1RgbaPath).string()) << "\",\n";
        os << "      \"elem_type\": \"u8\",\n";
        os << "      \"elem_count\": " << decoded1.byteCount << "\n";
        os << "    },\n";
        os << "    \"image2_rgba8\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->image2RgbaPath).string()) << "\",\n";
        os << "      \"elem_type\": \"u8\",\n";
        os << "      \"elem_count\": " << decoded2.byteCount << "\n";
        os << "    },\n";
        os << "    \"stage0_dssim5x5_gaussian_linear_u32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage0DssimPath).string()) << "\",\n";
        os << "      \"elem_type\": \"u32_le\",\n";
        os << "      \"elem_count\": " << debugInfo->stage0ElemCount << "\n";
        os << "    },\n";
        os << "    \"stage0_mu1_f32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage0Mu1Path).string()) << "\",\n";
        os << "      \"elem_type\": \"f32_le\",\n";
        os << "      \"elem_count\": " << debugInfo->stage0ElemCount << "\n";
        os << "    },\n";
        os << "    \"stage0_mu2_f32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage0Mu2Path).string()) << "\",\n";
        os << "      \"elem_type\": \"f32_le\",\n";
        os << "      \"elem_count\": " << debugInfo->stage0ElemCount << "\n";
        os << "    },\n";
        os << "    \"stage0_var1_f32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage0Var1Path).string()) << "\",\n";
        os << "      \"elem_type\": \"f32_le\",\n";
        os << "      \"elem_count\": " << debugInfo->stage0ElemCount << "\n";
        os << "    },\n";
        os << "    \"stage0_var2_f32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage0Var2Path).string()) << "\",\n";
        os << "      \"elem_type\": \"f32_le\",\n";
        os << "      \"elem_count\": " << debugInfo->stage0ElemCount << "\n";
        os << "    },\n";
        os << "    \"stage0_cov12_f32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage0Cov12Path).string()) << "\",\n";
        os << "      \"elem_type\": \"f32_le\",\n";
        os << "      \"elem_count\": " << debugInfo->stage0ElemCount << "\n";
        os << "    },\n";
        os << "    \"stage0_dssim3x3_u32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage0DssimPath).string()) << "\",\n";
        os << "      \"elem_type\": \"u32_le\",\n";
        os << "      \"elem_count\": " << debugInfo->stage0ElemCount << "\n";
        os << "    },\n";
        os << "    \"stage0_absdiff_u32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage0DssimPath).string()) << "\",\n";
        os << "      \"elem_type\": \"u32_le\",\n";
        os << "      \"elem_count\": " << debugInfo->stage0ElemCount << "\n";
        os << "    }";
        if (debugInfo->stage1ElemCount > 0) {
            os << ",\n";
            os << "    \"image1_scale1_rgba8\": {\n";
            os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->image1Scale1Path).string()) << "\",\n";
            os << "      \"elem_type\": \"u8\",\n";
            os << "      \"elem_count\": " << (debugInfo->stage1ElemCount * 4u) << "\n";
            os << "    },\n";
            os << "    \"image2_scale1_rgba8\": {\n";
            os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->image2Scale1Path).string()) << "\",\n";
            os << "      \"elem_type\": \"u8\",\n";
            os << "      \"elem_count\": " << (debugInfo->stage1ElemCount * 4u) << "\n";
            os << "    },\n";
            os << "    \"stage1_dssim5x5_gaussian_linear_u32le\": {\n";
            os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage1DssimPath).string()) << "\",\n";
            os << "      \"elem_type\": \"u32_le\",\n";
            os << "      \"elem_count\": " << debugInfo->stage1ElemCount << "\n";
            os << "    }";
            os << "\n";
        } else {
            os << "\n";
        }
        os << "  }";
    }

    os << "\n";
    os << "}\n";
    return os.str();
}

void WriteStringFile(const std::filesystem::path& outPath, const std::string& content) {
    const auto parent = outPath.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream output(outPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open output: " + outPath.string());
    }

    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!output) {
        throw std::runtime_error("failed to write output: " + outPath.string());
    }
}

void VkCheck(VkResult result, const std::string_view operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with VkResult " +
            std::to_string(static_cast<int>(result)));
    }
}

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1u) / alignment) * alignment;
}

std::vector<std::uint32_t> ReadSpirv(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open SPIR-V file: " + path.string());
    }
    const std::streamsize byteSize = input.tellg();
    if (byteSize <= 0 || (byteSize % 4) != 0) {
        throw std::runtime_error("invalid SPIR-V byte size: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint32_t> code(static_cast<std::size_t>(byteSize) / sizeof(std::uint32_t));
    if (!input.read(
            reinterpret_cast<char*>(code.data()),
            static_cast<std::streamsize>(code.size() * sizeof(std::uint32_t)))) {
        throw std::runtime_error("failed to read SPIR-V file: " + path.string());
    }
    return code;
}

GpuBuffer CreateBuffer(const GpuSession& session, std::uint64_t byteSize, gpu::MemoryType memory) {
    GpuBuffer result;
    result.heap = gpu::create_gpu_heap(session.gpuDevice, std::max<std::uint64_t>(byteSize, 4u), memory);
    if (!result.heap.owner) throw std::runtime_error("NoGraphicsAPI GPU allocation failed");
    result.size = result.heap.range.size;
    result.mapped = result.heap.range.cpu;
    return result;
}

GpuSession::~GpuSession() {
    if (!gpuDevice) return;
    if (commands) {
        gpu::discard_vulkan_commands(commands);
        commands = nullptr;
    }
    gpu::wait_idle(gpuDevice);
    debugComputeArenas.reset();
    batchComputeArenas.reset();
    srgbToLinearLutBuffer.Reset();
    gpu::destroy_pso(rgba8ToLinearShader);
    gpu::destroy_pso(preprocessShader);
    gpu::destroy_pso(stage0Shader);
    gpu::destroy_pso(stage0ScoreShader);
    gpu::destroy_pso(reduceSumShader);
    gpu::destroy_pso(reduceAbsDeviationShader);
    gpu::destroy_pso(downsampleShader);
    gpu::destroy_pso(vulkanYuvToRgbaShader);
    if (videoTextureHeap.owner) gpu::destroy_gpu_heap(videoTextureHeap);
    if (videoSamplerHeap.owner) gpu::destroy_gpu_heap(videoSamplerHeap);
    if (timestampQueryPool) vkDestroyQueryPool(device, timestampQueryPool, nullptr);
    if (completion) gpu::destroy_timeline_semaphore(completion);
    gpu::destroy_device(gpuDevice);
}

void BeginCommands(GpuSession& session) {
    session.commands = gpu::begin_commands(session.gpuDevice);
    session.commandBuffer = gpu::get_vulkan_commands(session.commands);
    if (session.timestampQueryEnabled)
        vkCmdResetQueryPool(session.commandBuffer, session.timestampQueryPool, 0, 2);
}

void BeginTimestamp(GpuSession& session) {
    if (session.timestampQueryEnabled) {
        vkCmdWriteTimestamp2(
            session.commandBuffer,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            session.timestampQueryPool,
            0);
    }
}

void EndTimestamp(GpuSession& session) {
    if (session.timestampQueryEnabled) {
        vkCmdWriteTimestamp2(
            session.commandBuffer,
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
            session.timestampQueryPool,
            1);
    }
}

void SubmitCommands(
    GpuSession& session,
    std::span<const VulkanVideoFrame* const> videoFrames = {}) {
    std::array<VkSemaphoreSubmitInfo, 2> waits{}, signals{};
    std::array<AVVkFrame*, 2> owners{};
    std::uint32_t count = 0;
    for (const VulkanVideoFrame* frame : videoFrames) {
        if (!frame || !frame->vkFrame || !frame->vkFrame->sem[0]) continue;
        const VkSemaphore semaphore = frame->vkFrame->sem[0];
        bool duplicate = false;
        for (std::uint32_t i = 0; i < count; ++i)
            duplicate |= waits[i].semaphore == semaphore;
        if (duplicate) continue;
        if (count == waits.size()) throw std::runtime_error("too many video semaphores");
        waits[count] = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = semaphore,
            .value = frame->vkFrame->sem_value[0],
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
        signals[count] = waits[count];
        ++signals[count].value;
        owners[count++] = frame->vkFrame;
    }
    const gpu::TimelinePoint completion{session.completion, ++session.completionValue};
    if (count) {
        gpu::submit_vulkan({&session.commands, 1}, completion,
                           {waits.data(), count}, {signals.data(), count});
    } else {
        gpu::submit({&session.commands, 1}, completion);
    }
    session.commands = nullptr;
    session.commandBuffer = VK_NULL_HANDLE;
    for (std::uint32_t i = 0; i < count; ++i) ++owners[i]->sem_value[0];
}

void WaitForSubmission(GpuSession& session) {
    gpu::wait_timeline({session.completion, session.completionValue});
}

void SetBufferAddresses(GpuSession& session,
                        std::span<const gpu::GpuRange> ranges,
                        std::uint32_t firstBinding = 0) {
    if (firstBinding + ranges.size() > std::size(session.root.buffers))
        throw std::runtime_error("too many GPU addresses in compute root");
    for (std::size_t i = 0; i < ranges.size(); ++i)
        session.root.buffers[firstBinding + i] = static_cast<std::uint32_t*>(ranges[i].gpu);
}

template <typename Params>
void SetParams(GpuSession& session, const Params& params) {
    static_assert(sizeof(Params) == sizeof(session.root.params));
    std::memcpy(session.root.params, &params, sizeof(params));
}

struct MemoryCopy {
    std::uint64_t srcOffset = 0;
    std::uint64_t dstOffset = 0;
    std::uint64_t size = 0;
};

void CopyRegions(GpuSession& session, const GpuBuffer& source, const GpuBuffer& destination,
                 std::uint32_t count, const MemoryCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& region = regions[i];
        if (region.srcOffset > source.size || region.size > source.size - region.srcOffset ||
            region.dstOffset > destination.size || region.size > destination.size - region.dstOffset)
            throw std::runtime_error("GPU copy exceeds its allocation");
        gpu::copy_memory(session.commands,
            {reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(source.heap.range.gpu) + region.srcOffset), region.size},
            {reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(destination.heap.range.gpu) + region.dstOffset), region.size});
    }
}

double ReadGpuTimestampMs(GpuSession& session) {
    if (!session.timestampQueryEnabled) {
        return 0.0;
    }
    std::array<std::uint64_t, 2> timestamps{};
    VkCheck(
        vkGetQueryPoolResults(
            session.device,
            session.timestampQueryPool,
            0,
            2,
            sizeof(timestamps),
            timestamps.data(),
            sizeof(std::uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
        "vkGetQueryPoolResults");
    const std::uint64_t validMask =
        (session.timestampValidBits >= 64u)
            ? std::numeric_limits<std::uint64_t>::max()
            : ((std::uint64_t{1} << session.timestampValidBits) - 1u);
    const std::uint64_t elapsedTicks =
        (timestamps[1] - timestamps[0]) & validMask;
    return static_cast<double>(elapsedTicks) *
           static_cast<double>(session.physicalDeviceProperties.limits.timestampPeriod) /
           1'000'000.0;
}

double SumF32(const float* values, std::size_t valueCount) {
    std::array<double, 8> sums{};
    std::size_t i = 0;
    for (; i + sums.size() <= valueCount; i += sums.size()) {
        sums[0] += static_cast<double>(values[i + 0]);
        sums[1] += static_cast<double>(values[i + 1]);
        sums[2] += static_cast<double>(values[i + 2]);
        sums[3] += static_cast<double>(values[i + 3]);
        sums[4] += static_cast<double>(values[i + 4]);
        sums[5] += static_cast<double>(values[i + 5]);
        sums[6] += static_cast<double>(values[i + 6]);
        sums[7] += static_cast<double>(values[i + 7]);
    }
    double sum = std::accumulate(sums.begin(), sums.end(), 0.0);
    for (; i < valueCount; ++i) {
        sum += static_cast<double>(values[i]);
    }
    return sum;
}

double SumAbsoluteDeviation(
    const float* values,
    std::size_t valueCount,
    double average) {
    std::array<double, 8> sums{};
    std::size_t i = 0;
    for (; i + sums.size() <= valueCount; i += sums.size()) {
        sums[0] += std::abs(average - static_cast<double>(values[i + 0]));
        sums[1] += std::abs(average - static_cast<double>(values[i + 1]));
        sums[2] += std::abs(average - static_cast<double>(values[i + 2]));
        sums[3] += std::abs(average - static_cast<double>(values[i + 3]));
        sums[4] += std::abs(average - static_cast<double>(values[i + 4]));
        sums[5] += std::abs(average - static_cast<double>(values[i + 5]));
        sums[6] += std::abs(average - static_cast<double>(values[i + 6]));
        sums[7] += std::abs(average - static_cast<double>(values[i + 7]));
    }
    double sum = std::accumulate(sums.begin(), sums.end(), 0.0);
    for (; i < valueCount; ++i) {
        sum += std::abs(average - static_cast<double>(values[i]));
    }
    return sum;
}

struct BufferRegion {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

struct ArenaBuilder {
    std::uint64_t cursor = 0;
    std::uint64_t alignment = 4;

    BufferRegion Add(std::uint64_t byteSize) {
        cursor = AlignUp(cursor, alignment);
        const BufferRegion region = {.offset = cursor, .size = byteSize};
        cursor += byteSize;
        return region;
    }
};

gpu::GpuRange DescribeBuffer(const GpuBuffer& buffer, const BufferRegion& region) {
    if (region.offset > buffer.size || region.size > buffer.size - region.offset)
        throw std::runtime_error("GPU address range exceeds its allocation");
    return {
        .gpu = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(buffer.heap.range.gpu) + region.offset),
        .size = region.size,
    };
}

void ValidateStorageRange(const GpuSession&, std::uint64_t byteSize, const std::string_view label) {
    if (byteSize == 0) throw std::runtime_error(std::string(label) + " is empty");
}

std::array<std::uint32_t, 2> ComputeWorkgroupCounts(
    const GpuSession& session,
    std::uint32_t width,
    std::uint32_t height) {
    const std::uint64_t wgX = (static_cast<std::uint64_t>(width) + 15u) / 16u;
    const std::uint64_t wgY = (static_cast<std::uint64_t>(height) + 15u) / 16u;
    const auto& limits = session.physicalDeviceProperties.limits;
    if (wgX > limits.maxComputeWorkGroupCount[0] ||
        wgY > limits.maxComputeWorkGroupCount[1]) {
        throw std::runtime_error(
            "image dimensions exceed Vulkan compute workgroup-count limits");
    }
    return {
        static_cast<std::uint32_t>(wgX),
        static_cast<std::uint32_t>(wgY),
    };
}

struct DebugStageLayout {
    BufferRegion input1;
    BufferRegion input2;
    BufferRegion lab1;
    BufferRegion lab2;
    BufferRegion ssim;
    BufferRegion mu1;
    BufferRegion mu2;
    BufferRegion var1;
    BufferRegion var2;
    BufferRegion cov12;
    BufferRegion upload1;
    BufferRegion upload2;
    BufferRegion readbackSsim;
    BufferRegion readbackMu1;
    BufferRegion readbackMu2;
    BufferRegion readbackVar1;
    BufferRegion readbackVar2;
    BufferRegion readbackCov12;
    std::uint64_t workspaceBytes = 0;
    std::uint64_t uploadBytes = 0;
    std::uint64_t readbackBytes = 0;
};

DebugStageLayout BuildDebugStageLayout(
    const GpuSession& session,
    std::size_t elemCount,
    bool includeStats) {
    const std::uint64_t rgbaBytes = elemCount * sizeof(LinearRgba);
    const std::uint64_t f32Bytes = elemCount * sizeof(float);
    ValidateStorageRange(session, rgbaBytes, "debug RGBA/LAB buffer");
    ValidateStorageRange(session, f32Bytes, "debug SSIM/statistics buffer");
    ArenaBuilder workspace{
        .alignment = std::max<std::uint64_t>(
            4u,
            16u),
    };
    DebugStageLayout layout;
    layout.input1 = workspace.Add(rgbaBytes);
    layout.input2 = workspace.Add(rgbaBytes);
    layout.lab1 = workspace.Add(rgbaBytes);
    layout.lab2 = workspace.Add(rgbaBytes);
    layout.ssim = workspace.Add(f32Bytes);
    if (includeStats) {
        layout.mu1 = workspace.Add(f32Bytes);
        layout.mu2 = workspace.Add(f32Bytes);
        layout.var1 = workspace.Add(f32Bytes);
        layout.var2 = workspace.Add(f32Bytes);
        layout.cov12 = workspace.Add(f32Bytes);
    }
    layout.workspaceBytes = workspace.cursor;

    ArenaBuilder upload{.alignment = 4u};
    layout.upload1 = upload.Add(rgbaBytes);
    layout.upload2 = upload.Add(rgbaBytes);
    layout.uploadBytes = upload.cursor;

    ArenaBuilder readback{.alignment = 4u};
    layout.readbackSsim = readback.Add(f32Bytes);
    if (includeStats) {
        layout.readbackMu1 = readback.Add(f32Bytes);
        layout.readbackMu2 = readback.Add(f32Bytes);
        layout.readbackVar1 = readback.Add(f32Bytes);
        layout.readbackVar2 = readback.Add(f32Bytes);
        layout.readbackCov12 = readback.Add(f32Bytes);
    }
    layout.readbackBytes = readback.cursor;
    return layout;
}

ScaleOutputs RunStage0Compute(
    GpuSession& session,
    const std::vector<LinearRgba>& input1,
    const std::vector<LinearRgba>& input2,
    std::uint32_t width,
    std::uint32_t height,
    std::size_t scaleLevel,
    bool readIntermediateStats) {
    if (input1.size() != input2.size()) {
        throw std::runtime_error("input buffer size mismatch");
    }
    if (input1.empty()) {
        return {};
    }
    const std::size_t elemCount = input1.size();
    const std::size_t expectedCount =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (expectedCount != elemCount || elemCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("pixel count mismatch or input too large");
    }
    const DebugStageLayout layout =
        BuildDebugStageLayout(session, elemCount, readIntermediateStats);
    const std::uint64_t rgbaBytes = elemCount * sizeof(LinearRgba);
    const std::uint64_t f32Bytes = elemCount * sizeof(float);

    ScaleOutputs outputs;
    if (!session.debugComputeArenas ||
        session.debugComputeArenas->workspaceCapacity < layout.workspaceBytes ||
        session.debugComputeArenas->uploadCapacity < layout.uploadBytes ||
        session.debugComputeArenas->readbackCapacity < layout.readbackBytes) {
        const auto startedAt = std::chrono::steady_clock::now();
        auto resources = std::make_unique<ComputeArenas>();
        resources->workspaceCapacity = layout.workspaceBytes;
        resources->uploadCapacity = layout.uploadBytes;
        resources->readbackCapacity = layout.readbackBytes;
        resources->workspace = CreateBuffer(
            session,
            layout.workspaceBytes,
            gpu::MemoryType::gpu_only);
        resources->upload = CreateBuffer(
            session,
            layout.uploadBytes,
            gpu::MemoryType::cpu_visible);
        resources->readback = CreateBuffer(
            session,
            layout.readbackBytes,
            gpu::MemoryType::readback);
        session.debugComputeArenas = std::move(resources);
        outputs.createBuffers_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startedAt);
    }
    ComputeArenas& resources = *session.debugComputeArenas;

    const auto writeStartedAt = std::chrono::steady_clock::now();
    auto* uploadBytes = static_cast<std::uint8_t*>(resources.upload.mapped);
    std::memcpy(uploadBytes + layout.upload1.offset, input1.data(), rgbaBytes);
    std::memcpy(uploadBytes + layout.upload2.offset, input2.data(), rgbaBytes);

    outputs.writeInputBuffers_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - writeStartedAt);

    struct ParamsData {
        std::uint32_t len;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t qscale;
    };
    const ParamsData params = {
        .len = static_cast<std::uint32_t>(elemCount),
        .width = width,
        .height = height,
        .qscale = kStage0QScale,
    };
    const auto workgroups = ComputeWorkgroupCounts(session, width, height);
    const std::uint32_t wgX = workgroups[0];
    const std::uint32_t wgY = workgroups[1];

    const auto dispatchStartedAt = std::chrono::steady_clock::now();
    BeginCommands(session);
    const std::array<MemoryCopy, 2> uploads = {{
        {.srcOffset = layout.upload1.offset, .dstOffset = layout.input1.offset, .size = rgbaBytes},
        {.srcOffset = layout.upload2.offset, .dstOffset = layout.input2.offset, .size = rgbaBytes},
    }};
    CopyRegions(session, resources.upload, resources.workspace,
        static_cast<std::uint32_t>(uploads.size()),
        uploads.data());
    gpu::barrier(session.commands,
        gpu::Stage::transfer,
        gpu::Access::transfer_write,
        gpu::Stage::compute,
        gpu::Access::shader_read);
    BeginTimestamp(session);

    gpu::bind_pso(session.commands, session.preprocessShader);
    const std::array<gpu::GpuRange, 2> preprocess1 = {{
        DescribeBuffer(resources.workspace, layout.input1),
        DescribeBuffer(resources.workspace, layout.lab1),
    }};
    SetBufferAddresses(session, preprocess1);
    SetParams(session, params);
    gpu::dispatch(session.commands, session.root, {.x = wgX, .y = wgY, .z = 1});
    const std::array<gpu::GpuRange, 2> preprocess2 = {{
        DescribeBuffer(resources.workspace, layout.input2),
        DescribeBuffer(resources.workspace, layout.lab2),
    }};
    SetBufferAddresses(session, preprocess2);
    SetParams(session, params);
    gpu::dispatch(session.commands, session.root, {.x = wgX, .y = wgY, .z = 1});
    gpu::barrier(session.commands,
        gpu::Stage::compute,
        gpu::Access::shader_write,
        gpu::Stage::compute,
        gpu::Access::shader_read);

    if (readIntermediateStats) {
        gpu::bind_pso(session.commands, session.stage0Shader);
        const std::array<gpu::GpuRange, 8> descriptors = {{
            DescribeBuffer(resources.workspace, layout.lab1),
            DescribeBuffer(resources.workspace, layout.lab2),
            DescribeBuffer(resources.workspace, layout.ssim),
            DescribeBuffer(resources.workspace, layout.mu1),
            DescribeBuffer(resources.workspace, layout.mu2),
            DescribeBuffer(resources.workspace, layout.var1),
            DescribeBuffer(resources.workspace, layout.var2),
            DescribeBuffer(resources.workspace, layout.cov12),
        }};
        SetBufferAddresses(session, descriptors);
        SetParams(session, params);
    } else {
        gpu::bind_pso(session.commands, session.stage0ScoreShader);
        const std::array<gpu::GpuRange, 3> descriptors = {{
            DescribeBuffer(resources.workspace, layout.lab1),
            DescribeBuffer(resources.workspace, layout.lab2),
            DescribeBuffer(resources.workspace, layout.ssim),
        }};
        SetBufferAddresses(session, descriptors);
        SetParams(session, params);
    }
    gpu::dispatch(session.commands, session.root, {.x = wgX, .y = wgY, .z = 1});
    EndTimestamp(session);
    gpu::barrier(session.commands,
        gpu::Stage::compute,
        gpu::Access::shader_write,
        gpu::Stage::transfer,
        gpu::Access::transfer_read);

    std::array<MemoryCopy, 6> readbacks{};
    std::uint32_t readbackCount = 1;
    readbacks[0] = {
        .srcOffset = layout.ssim.offset,
        .dstOffset = layout.readbackSsim.offset,
        .size = f32Bytes,
    };
    if (readIntermediateStats) {
        const std::array<BufferRegion, 5> sources = {
            layout.mu1, layout.mu2, layout.var1, layout.var2, layout.cov12};
        const std::array<BufferRegion, 5> destinations = {
            layout.readbackMu1,
            layout.readbackMu2,
            layout.readbackVar1,
            layout.readbackVar2,
            layout.readbackCov12,
        };
        for (std::size_t i = 0; i < sources.size(); ++i) {
            readbacks[i + 1u] = {
                .srcOffset = sources[i].offset,
                .dstOffset = destinations[i].offset,
                .size = f32Bytes,
            };
        }
        readbackCount = static_cast<std::uint32_t>(readbacks.size());
    }
    CopyRegions(session, resources.workspace, resources.readback,
        readbackCount,
        readbacks.data());
    gpu::barrier(session.commands,
        gpu::Stage::transfer,
        gpu::Access::transfer_write,
        gpu::Stage::host,
        gpu::Access::host_read);
    SubmitCommands(session);
    outputs.dispatchAndSubmit_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - dispatchStartedAt);

    outputs.width = width;
    outputs.height = height;
    outputs.elemCount = elemCount;
    const auto readbackStartedAt = std::chrono::steady_clock::now();
    WaitForSubmission(session);

    const auto* mapped = static_cast<const std::uint8_t*>(resources.readback.mapped);
    outputs.ssimMap.resize(elemCount);
    std::memcpy(outputs.ssimMap.data(), mapped + layout.readbackSsim.offset, f32Bytes);
    if (readIntermediateStats) {
        outputs.mu1.resize(elemCount);
        outputs.mu2.resize(elemCount);
        outputs.var1.resize(elemCount);
        outputs.var2.resize(elemCount);
        outputs.cov12.resize(elemCount);
        std::memcpy(outputs.mu1.data(), mapped + layout.readbackMu1.offset, f32Bytes);
        std::memcpy(outputs.mu2.data(), mapped + layout.readbackMu2.offset, f32Bytes);
        std::memcpy(outputs.var1.data(), mapped + layout.readbackVar1.offset, f32Bytes);
        std::memcpy(outputs.var2.data(), mapped + layout.readbackVar2.offset, f32Bytes);
        std::memcpy(outputs.cov12.data(), mapped + layout.readbackCov12.offset, f32Bytes);
    }
    outputs.gpuTimestampMs = ReadGpuTimestampMs(session);
    outputs.readback_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - readbackStartedAt);

    const auto postStartedAt = std::chrono::steady_clock::now();
    const double ssimSum = SumF32(outputs.ssimMap.data(), elemCount);
    outputs.meanSsim = ssimSum / static_cast<double>(elemCount);
    const double avg =
        std::pow(std::max(outputs.meanSsim, 0.0), std::pow(0.5, static_cast<double>(scaleLevel)));
    const double devSum = SumAbsoluteDeviation(outputs.ssimMap.data(), elemCount, avg);
    outputs.ssimScore = 1.0 - devSum / static_cast<double>(elemCount);
    outputs.postProcess_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - postStartedAt);
    return outputs;
}
struct BatchStageLayout {
    BufferRegion rgba8Input1;
    BufferRegion rgba8Input2;
    BufferRegion input1;
    BufferRegion input2;
    BufferRegion downsample1;
    BufferRegion downsample2;
    BufferRegion lab1;
    BufferRegion lab2;
    BufferRegion ssim;
    BufferRegion reductionScratch1;
    BufferRegion reductionScratch2;
    std::array<BufferRegion, kDefaultScaleWeights.size()> ssimSums;
    std::array<BufferRegion, kDefaultScaleWeights.size()> deviationSums;
    BufferRegion upload1;
    BufferRegion upload2;
    std::uint64_t workspaceBytes = 0;
    std::uint64_t uploadBytes = 0;
};

BatchStageLayout BuildBatchStageLayout(
    const GpuSession&,
    std::uint64_t rgba8Bytes,
    std::uint64_t inputBytes,
    std::uint64_t downsampleBytes,
    std::uint64_t f32Bytes,
    std::uint64_t reductionScratchBytes,
    std::size_t levelCount) {
    ArenaBuilder workspace{
        .alignment = std::max<std::uint64_t>(
            4u,
            16u),
    };
    BatchStageLayout layout;
    layout.rgba8Input1 = workspace.Add(rgba8Bytes);
    layout.rgba8Input2 = workspace.Add(rgba8Bytes);
    layout.input1 = workspace.Add(inputBytes);
    layout.input2 = workspace.Add(inputBytes);
    layout.downsample1 = workspace.Add(downsampleBytes);
    layout.downsample2 = workspace.Add(downsampleBytes);
    layout.lab1 = workspace.Add(inputBytes);
    layout.lab2 = workspace.Add(inputBytes);
    layout.ssim = workspace.Add(f32Bytes);
    layout.reductionScratch1 = workspace.Add(reductionScratchBytes);
    layout.reductionScratch2 = workspace.Add(reductionScratchBytes);
    for (std::size_t level = 0; level < levelCount; ++level) {
        layout.ssimSums[level] = workspace.Add(sizeof(float));
        layout.deviationSums[level] = workspace.Add(sizeof(float));
    }
    layout.workspaceBytes = workspace.cursor;

    ArenaBuilder upload{.alignment = 4u};
    layout.upload1 = upload.Add(rgba8Bytes);
    layout.upload2 = upload.Add(rgba8Bytes);
    layout.uploadBytes = upload.cursor;
    return layout;
}

struct VideoImageViews {
    std::uint32_t y = 0;
    std::uint32_t uv = 0;
    VkImageLayout originalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags originalAccess = 0;
    std::uint32_t originalQueueFamily = VK_QUEUE_FAMILY_IGNORED;
};

void WriteVideoPlaneDescriptor(const GpuSession& session, VkImage image,
                              VkFormat format, VkImageAspectFlags aspect, std::uint32_t slot) {
    const VkImageViewUsageCreateInfo usage{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
    };
    const VkImageViewCreateInfo view{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = &usage,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {.aspectMask = aspect, .levelCount = 1, .layerCount = 1},
    };
    const VkImageDescriptorInfoEXT imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
        .pView = &view,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkResourceDescriptorInfoEXT descriptor{
        .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
        .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .data = {.pImage = &imageInfo},
    };
    const auto stride = gpu::get_device_caps(session.gpuDevice).texture_descriptor_size;
    const VkHostAddressRangeEXT destination{
        .address = session.videoTextureHeap.range.cpu + slot * stride,
        .size = static_cast<std::size_t>(stride),
    };
    VkCheck(session.writeResourceDescriptors(session.device, 1, &descriptor, &destination),
            "vkWriteResourceDescriptorsEXT(video)");
}

VideoImageViews CreateVideoImageViews(const GpuSession& session, const VulkanVideoFrame& frame,
                                     std::uint32_t firstSlot) {
    if (!frame.vkFrame || !frame.framesContext)
        throw std::runtime_error("invalid Vulkan Video frame metadata");
    if (frame.vkFrame->img[1] != VK_NULL_HANDLE)
        throw std::runtime_error("separate-plane Vulkan Video frames are not supported");
    const bool tenBit = frame.softwareFormat == AV_PIX_FMT_P010LE ||
                        frame.softwareFormat == AV_PIX_FMT_P010BE;
    VideoImageViews views{
        .y = firstSlot,
        .uv = firstSlot + 1,
        .originalLayout = frame.vkFrame->layout[0],
        .originalAccess = static_cast<VkAccessFlags>(frame.vkFrame->access[0]),
        .originalQueueFamily = frame.vkFrame->queue_family[0],
    };
    WriteVideoPlaneDescriptor(session, frame.vkFrame->img[0],
        tenBit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_PLANE_0_BIT, views.y);
    WriteVideoPlaneDescriptor(session, frame.vkFrame->img[0],
        tenBit ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R8G8_UNORM, VK_IMAGE_ASPECT_PLANE_1_BIT, views.uv);
    return views;
}

class VideoFrameLock {
public:
    explicit VideoFrameLock(const VulkanVideoFrame* frame) : frame_(frame) {
        if (!frame_) return;
        context_ = reinterpret_cast<AVVulkanFramesContext*>(frame_->framesContext->hwctx);
        context_->lock_frame(frame_->framesContext, frame_->vkFrame);
        layout_ = frame_->vkFrame->layout[0];
        access_ = frame_->vkFrame->access[0];
        queue_ = frame_->vkFrame->queue_family[0];
    }
    ~VideoFrameLock() {
        if (!frame_) return;
        frame_->vkFrame->layout[0] = layout_;
        frame_->vkFrame->access[0] = access_;
        frame_->vkFrame->queue_family[0] = queue_;
        Unlock();
    }
    VideoFrameLock(const VideoFrameLock&) = delete;
    VideoFrameLock& operator=(const VideoFrameLock&) = delete;
    void Unlock() {
        if (frame_) {
            context_->unlock_frame(frame_->framesContext, frame_->vkFrame);
            frame_ = nullptr;
        }
    }
private:
    const VulkanVideoFrame* frame_ = nullptr;
    AVVulkanFramesContext* context_ = nullptr;
    VkImageLayout layout_{};
    VkAccessFlagBits access_{};
    std::uint32_t queue_ = VK_QUEUE_FAMILY_IGNORED;
};

void RecordVideoImageBarrier(
    GpuSession& session,
    const VulkanVideoFrame& frame,
    VkImageLayout newLayout,
    VkAccessFlags2 newAccess,
    std::uint32_t newQueueFamily,
    VkPipelineStageFlags2 srcStage,
    VkPipelineStageFlags2 dstStage) {
    if (frame.vkFrame->img[1] != VK_NULL_HANDLE) {
        throw std::runtime_error("separate-plane Vulkan Video frames are not supported");
    }
    const std::uint32_t destinationQueueFamily =
        frame.vkFrame->queue_family[0] == VK_QUEUE_FAMILY_IGNORED
            ? VK_QUEUE_FAMILY_IGNORED
            : newQueueFamily;
    const VkImageMemoryBarrier2 imageBarrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = srcStage,
        .srcAccessMask = static_cast<VkAccessFlags2>(frame.vkFrame->access[0]),
        .dstStageMask = dstStage,
        .dstAccessMask = newAccess,
        .oldLayout = frame.vkFrame->layout[0],
        .newLayout = newLayout,
        .srcQueueFamilyIndex = frame.vkFrame->queue_family[0],
        .dstQueueFamilyIndex = destinationQueueFamily,
        .image = frame.vkFrame->img[0],
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    const VkDependencyInfo dependency = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &imageBarrier,
    };
    vkCmdPipelineBarrier2(session.commandBuffer, &dependency);
    frame.vkFrame->layout[0] = newLayout;
    frame.vkFrame->access[0] = static_cast<VkAccessFlagBits>(newAccess);
    frame.vkFrame->queue_family[0] = destinationQueueFamily;
}

std::vector<ScaleOutputs> RunStage0BatchCompute(
    GpuSession& session,
    std::span<const std::uint8_t> input1,
    std::span<const std::uint8_t> input2,
    const std::vector<std::uint32_t>& widths,
    const std::vector<std::uint32_t>& heights,
    const VulkanVideoFrame* video1 = nullptr,
    const VulkanVideoFrame* video2 = nullptr) {
    const std::size_t levelCount = widths.size();
    if (levelCount == 0 || levelCount > kDefaultScaleWeights.size() ||
        heights.size() != levelCount) {
        throw std::runtime_error("invalid batch dimensions");
    }

    std::vector<std::uint64_t> sumOutputOffsets(levelCount);
    std::vector<std::uint64_t> deviationOutputOffsets(levelCount);
    std::vector<std::size_t> elemCounts(levelCount);
    for (std::size_t level = 0; level < levelCount; ++level) {
        const std::size_t elemCount =
            static_cast<std::size_t>(widths[level]) * static_cast<std::size_t>(heights[level]);
        if (elemCount == 0 || elemCount > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("invalid or oversized pyramid level");
        }
        elemCounts[level] = elemCount;
        sumOutputOffsets[level] = level * 2u * sizeof(float);
        deviationOutputOffsets[level] = sumOutputOffsets[level] + sizeof(float);
    }
    const std::uint64_t outputBytesTotal = levelCount * 2u * sizeof(float);

    const std::size_t baseElemCount = elemCounts.front();
    const std::uint64_t rgba8Bytes = baseElemCount * 4u;
    const bool videoInput = video1 != nullptr || video2 != nullptr;
    if (videoInput && (video1 == nullptr || video2 == nullptr)) {
        throw std::runtime_error("both video inputs must be Vulkan frames");
    }
    if ((!videoInput && (input1.size() != rgba8Bytes || input2.size() != rgba8Bytes)) ||
        (videoInput &&
         (video1->width != widths.front() || video1->height != heights.front() ||
          video2->width != widths.front() || video2->height != heights.front()))) {
        throw std::runtime_error("batch input dimensions do not match pixel count");
    }
    const std::uint64_t inputBytes = baseElemCount * sizeof(LinearRgba);
    const std::uint64_t downsampleBytes =
        ((levelCount > 1u) ? elemCounts[1] : 1u) * sizeof(LinearRgba);
    const std::uint64_t baseF32Bytes = baseElemCount * sizeof(float);
    const auto baseWorkgroups =
        ComputeWorkgroupCounts(session, widths.front(), heights.front());
    const std::uint64_t reductionScratchBytes =
        static_cast<std::uint64_t>(baseWorkgroups[0]) * baseWorkgroups[1] * sizeof(float);
    ValidateStorageRange(session, rgba8Bytes, "packed RGBA8 input buffer");
    ValidateStorageRange(session, inputBytes, "linear RGBA/LAB buffer");
    ValidateStorageRange(session, downsampleBytes, "downsample buffer");
    ValidateStorageRange(session, baseF32Bytes, "SSIM output buffer");
    ValidateStorageRange(session, reductionScratchBytes, "reduction scratch buffer");
    const BatchStageLayout layout = BuildBatchStageLayout(
        session,
        rgba8Bytes,
        inputBytes,
        downsampleBytes,
        baseF32Bytes,
        reductionScratchBytes,
        levelCount);

    std::vector<ScaleOutputs> outputs(levelCount);
    if (!session.batchComputeArenas ||
        session.batchComputeArenas->workspaceCapacity < layout.workspaceBytes ||
        session.batchComputeArenas->uploadCapacity < layout.uploadBytes ||
        session.batchComputeArenas->readbackCapacity < outputBytesTotal) {
        const auto startedAt = std::chrono::steady_clock::now();
        auto resources = std::make_unique<ComputeArenas>();
        resources->workspaceCapacity = layout.workspaceBytes;
        resources->uploadCapacity = layout.uploadBytes;
        resources->readbackCapacity = outputBytesTotal;
        resources->workspace = CreateBuffer(
            session,
            layout.workspaceBytes,
            gpu::MemoryType::gpu_only);
        resources->upload = CreateBuffer(
            session,
            layout.uploadBytes,
            gpu::MemoryType::cpu_visible);
        resources->readback = CreateBuffer(
            session,
            outputBytesTotal,
            gpu::MemoryType::readback);
        session.batchComputeArenas = std::move(resources);
        outputs.front().createBuffers_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startedAt);
    }
    ComputeArenas& resources = *session.batchComputeArenas;

    const auto writeStartedAt = std::chrono::steady_clock::now();
    if (!videoInput) {
        auto* upload = static_cast<std::uint8_t*>(resources.upload.mapped);
        std::memcpy(upload + layout.upload1.offset, input1.data(), rgba8Bytes);
        std::memcpy(upload + layout.upload2.offset, input2.data(), rgba8Bytes);

    }
    outputs.front().writeInputBuffers_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - writeStartedAt);

    struct ParamsData {
        std::uint32_t len;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t qscale;
    };
    struct DownsampleParamsData {
        std::uint32_t inWidth;
        std::uint32_t inHeight;
        std::uint32_t outWidth;
        std::uint32_t outHeight;
    };

    const auto dispatchStartedAt = std::chrono::steady_clock::now();
    std::unique_ptr<VideoImageViews> videoViews1;
    std::unique_ptr<VideoImageViews> videoViews2;
    if (videoInput) {
        videoViews1 = std::make_unique<VideoImageViews>(CreateVideoImageViews(session, *video1, 0));
        videoViews2 = std::make_unique<VideoImageViews>(CreateVideoImageViews(session, *video2, 2));
    }
    VideoFrameLock lock1(videoInput ? video1 : nullptr);
    VideoFrameLock lock2(videoInput ? video2 : nullptr);
    BeginCommands(session);
    if (!videoInput) {
        const std::array<MemoryCopy, 2> inputCopies = {{
            {
                .srcOffset = layout.upload1.offset,
                .dstOffset = layout.rgba8Input1.offset,
                .size = rgba8Bytes,
            },
            {
                .srcOffset = layout.upload2.offset,
                .dstOffset = layout.rgba8Input2.offset,
                .size = rgba8Bytes,
            },
        }};
        CopyRegions(session, resources.upload, resources.workspace,
            static_cast<std::uint32_t>(inputCopies.size()),
            inputCopies.data());
        gpu::barrier(session.commands,
            gpu::Stage::transfer,
            gpu::Access::transfer_write,
            gpu::Stage::compute,
            gpu::Access::shader_read);
    } else {
        RecordVideoImageBarrier(
            session,
            *video1,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_2_SHADER_READ_BIT,
            session.queueFamilyIndex,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        RecordVideoImageBarrier(
            session,
            *video2,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_2_SHADER_READ_BIT,
            session.queueFamilyIndex,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }
    BeginTimestamp(session);

    const ParamsData baseParams = {
        .len = static_cast<std::uint32_t>(baseElemCount),
        .width = widths.front(),
        .height = heights.front(),
        .qscale = kStage0QScale,
    };
    const std::uint32_t baseWgX = baseWorkgroups[0];
    const std::uint32_t baseWgY = baseWorkgroups[1];
    const gpu::GpuRange lutDescriptor = gpu::gpu_range(session.srgbToLinearLutBuffer.heap);
    if (!videoInput) {
        gpu::bind_pso(session.commands, session.rgba8ToLinearShader);
        const std::array<gpu::GpuRange, 3> convert1 = {{
            DescribeBuffer(resources.workspace, layout.rgba8Input1),
            DescribeBuffer(resources.workspace, layout.input1),
            lutDescriptor,
        }};
        SetBufferAddresses(session, convert1);
        SetParams(session, baseParams);
        gpu::dispatch(session.commands, session.root, {.x = baseWgX, .y = baseWgY, .z = 1});
        const std::array<gpu::GpuRange, 3> convert2 = {{
            DescribeBuffer(resources.workspace, layout.rgba8Input2),
            DescribeBuffer(resources.workspace, layout.input2),
            lutDescriptor,
        }};
        SetBufferAddresses(session, convert2);
        SetParams(session, baseParams);
        gpu::dispatch(session.commands, session.root, {.x = baseWgX, .y = baseWgY, .z = 1});
    } else {
        const struct YuvParams {
            std::uint32_t width;
            std::uint32_t height;
            std::uint32_t bitDepth;
            std::uint32_t fullRange;
        } yuvParams = {
            .width = widths.front(),
            .height = heights.front(),
            // FFmpeg's Vulkan plane views expose the normalized component
            // range used by the benchmark's NV12/P010 surfaces. Using the
            // 8-bit code range keeps both formats on the same conversion
            // path (and matches FFmpeg's limited-range conversion here).
            .bitDepth = 8u,
            .fullRange = (video1->colorRange == AVCOL_RANGE_JPEG ||
                          video2->colorRange == AVCOL_RANGE_JPEG) ? 1u : 0u,
        };
        gpu::set_texture_descriptor_heap(session.commands, gpu::gpu_range(session.videoTextureHeap));
        gpu::set_sampler_descriptor_heap(session.commands, gpu::gpu_range(session.videoSamplerHeap));
        gpu::bind_pso(session.commands, session.vulkanYuvToRgbaShader);
        session.root.yIndex = videoViews1->y;
        session.root.uvIndex = videoViews1->uv;
        const gpu::GpuRange output1 = DescribeBuffer(resources.workspace, layout.rgba8Input1);
        SetBufferAddresses(session, std::span(&output1, 1), 2);
        SetParams(session, yuvParams);
        gpu::dispatch(session.commands, session.root, {.x = baseWgX, .y = baseWgY, .z = 1});
        session.root.yIndex = videoViews2->y;
        session.root.uvIndex = videoViews2->uv;
        const gpu::GpuRange output2 = DescribeBuffer(resources.workspace, layout.rgba8Input2);
        SetBufferAddresses(session, std::span(&output2, 1), 2);
        SetParams(session, yuvParams);
        gpu::dispatch(session.commands, session.root, {.x = baseWgX, .y = baseWgY, .z = 1});
        RecordVideoImageBarrier(
            session, *video1, videoViews1->originalLayout,
            static_cast<VkAccessFlags2>(videoViews1->originalAccess),
            videoViews1->originalQueueFamily,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        RecordVideoImageBarrier(
            session, *video2, videoViews2->originalLayout,
            static_cast<VkAccessFlags2>(videoViews2->originalAccess),
            videoViews2->originalQueueFamily,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

        gpu::barrier(session.commands, gpu::Stage::compute, gpu::Access::shader_write,
                     gpu::Stage::compute, gpu::Access::shader_read);
        // Keep the same packed-RGBA8 -> linear-premultiplied path as still
        // images. The YUV conversion above only replaces the CPU upload.
        gpu::bind_pso(session.commands, session.rgba8ToLinearShader);
        const std::array<gpu::GpuRange, 3> convert1 = {{
            DescribeBuffer(resources.workspace, layout.rgba8Input1),
            DescribeBuffer(resources.workspace, layout.input1),
            lutDescriptor,
        }};
        SetBufferAddresses(session, convert1);
        SetParams(session, baseParams);
        gpu::dispatch(session.commands, session.root, {.x = baseWgX, .y = baseWgY, .z = 1});
        const std::array<gpu::GpuRange, 3> convert2 = {{
            DescribeBuffer(resources.workspace, layout.rgba8Input2),
            DescribeBuffer(resources.workspace, layout.input2),
            lutDescriptor,
        }};
        SetBufferAddresses(session, convert2);
        SetParams(session, baseParams);
        gpu::dispatch(session.commands, session.root, {.x = baseWgX, .y = baseWgY, .z = 1});
    }
    gpu::barrier(session.commands,
        gpu::Stage::compute,
        gpu::Access::shader_write,
        gpu::Stage::compute,
        gpu::Access::shader_read);

    const auto recordSumReduction = [&](BufferRegion inputRegion,
                                        std::uint32_t inputWidth,
                                        std::uint32_t inputHeight,
                                        const BufferRegion& finalOutput) {
        bool writeScratch1 =
            inputRegion.offset != layout.reductionScratch1.offset;
        for (;;) {
            const auto reductionWorkgroups =
                ComputeWorkgroupCounts(session, inputWidth, inputHeight);
            const std::uint32_t reductionWgX = reductionWorkgroups[0];
            const std::uint32_t reductionWgY = reductionWorkgroups[1];
            const bool finalPass = reductionWgX == 1u && reductionWgY == 1u;
            const std::uint64_t partialBytes =
                static_cast<std::uint64_t>(reductionWgX) * reductionWgY * sizeof(float);
            const BufferRegion outputRegion = finalPass
                                                  ? finalOutput
                                                  : BufferRegion{
                                                        (writeScratch1
                                                             ? layout.reductionScratch1.offset
                                                             : layout.reductionScratch2.offset),
                                                        partialBytes};
            const ParamsData reductionParams = {
                .len = inputWidth * inputHeight,
                .width = inputWidth,
                .height = inputHeight,
                .qscale = 0u,
            };
            gpu::bind_pso(session.commands, session.reduceSumShader);
            const std::array<gpu::GpuRange, 2> descriptors = {{
                DescribeBuffer(resources.workspace, inputRegion),
                DescribeBuffer(resources.workspace, outputRegion),
            }};
            SetBufferAddresses(session, descriptors);
            SetParams(session, reductionParams);
            gpu::dispatch(session.commands, session.root, {.x = reductionWgX, .y = reductionWgY, .z = 1});
            gpu::barrier(session.commands,
                gpu::Stage::compute,
                gpu::Access::shader_write,
                gpu::Stage::compute,
                gpu::Access::shader_read);
            if (finalPass) {
                break;
            }
            inputRegion = outputRegion;
            inputWidth = reductionWgX;
            inputHeight = reductionWgY;
            writeScratch1 = !writeScratch1;
        }
    };

    const auto recordDeviationReduction = [&](const BufferRegion& ssimRegion,
                                              std::uint32_t width,
                                              std::uint32_t height,
                                              std::uint32_t level,
                                              const BufferRegion& sumResult,
                                              const BufferRegion& deviationResult) {
        const auto workgroups = ComputeWorkgroupCounts(session, width, height);
        const std::uint32_t wgX = workgroups[0];
        const std::uint32_t wgY = workgroups[1];
        const bool finalPass = wgX == 1u && wgY == 1u;
        const std::uint64_t partialBytes =
            static_cast<std::uint64_t>(wgX) * wgY * sizeof(float);
        const BufferRegion outputRegion = finalPass
                                              ? deviationResult
                                              : BufferRegion{
                                                    layout.reductionScratch1.offset,
                                                    partialBytes};
        const ParamsData deviationParams = {
            .len = width * height,
            .width = width,
            .height = height,
            .qscale = level,
        };
        gpu::bind_pso(session.commands, session.reduceAbsDeviationShader);
        const std::array<gpu::GpuRange, 3> descriptors = {{
            DescribeBuffer(resources.workspace, ssimRegion),
            DescribeBuffer(resources.workspace, sumResult),
            DescribeBuffer(resources.workspace, outputRegion),
        }};
        SetBufferAddresses(session, descriptors);
        SetParams(session, deviationParams);
        gpu::dispatch(session.commands, session.root, {.x = wgX, .y = wgY, .z = 1});
        gpu::barrier(session.commands,
            gpu::Stage::compute,
            gpu::Access::shader_write,
            gpu::Stage::compute,
            gpu::Access::shader_read);
        if (!finalPass) {
            recordSumReduction(outputRegion, wgX, wgY, deviationResult);
        }
    };

    for (std::size_t level = 0; level < levelCount; ++level) {
        const std::uint64_t rgbaBytes = elemCounts[level] * sizeof(LinearRgba);
        const std::uint64_t f32Bytes = elemCounts[level] * sizeof(float);
        const BufferRegion currentInput1 = ((level & 1u) == 0u)
                                               ? BufferRegion{layout.input1.offset, rgbaBytes}
                                               : BufferRegion{layout.downsample1.offset, rgbaBytes};
        const BufferRegion currentInput2 = ((level & 1u) == 0u)
                                               ? BufferRegion{layout.input2.offset, rgbaBytes}
                                               : BufferRegion{layout.downsample2.offset, rgbaBytes};
        const BufferRegion currentLab1 = {layout.lab1.offset, rgbaBytes};
        const BufferRegion currentLab2 = {layout.lab2.offset, rgbaBytes};
        const BufferRegion currentSsim = {layout.ssim.offset, f32Bytes};
        const ParamsData params = {
            .len = static_cast<std::uint32_t>(elemCounts[level]),
            .width = widths[level],
            .height = heights[level],
            .qscale = kStage0QScale,
        };
        const auto workgroups =
            ComputeWorkgroupCounts(session, widths[level], heights[level]);
        const std::uint32_t wgX = workgroups[0];
        const std::uint32_t wgY = workgroups[1];

        gpu::bind_pso(session.commands, session.preprocessShader);
        const std::array<gpu::GpuRange, 2> preprocess1 = {{
            DescribeBuffer(resources.workspace, currentInput1),
            DescribeBuffer(resources.workspace, currentLab1),
        }};
        SetBufferAddresses(session, preprocess1);
        SetParams(session, params);
        gpu::dispatch(session.commands, session.root, {.x = wgX, .y = wgY, .z = 1});
        const std::array<gpu::GpuRange, 2> preprocess2 = {{
            DescribeBuffer(resources.workspace, currentInput2),
            DescribeBuffer(resources.workspace, currentLab2),
        }};
        SetBufferAddresses(session, preprocess2);
        SetParams(session, params);
        gpu::dispatch(session.commands, session.root, {.x = wgX, .y = wgY, .z = 1});
        gpu::barrier(session.commands,
            gpu::Stage::compute,
            gpu::Access::shader_write,
            gpu::Stage::compute,
            gpu::Access::shader_read);

        gpu::bind_pso(session.commands, session.stage0ScoreShader);
        const std::array<gpu::GpuRange, 3> scoreDescriptors = {{
            DescribeBuffer(resources.workspace, currentLab1),
            DescribeBuffer(resources.workspace, currentLab2),
            DescribeBuffer(resources.workspace, currentSsim),
        }};
        SetBufferAddresses(session, scoreDescriptors);
        SetParams(session, params);
        gpu::dispatch(session.commands, session.root, {.x = wgX, .y = wgY, .z = 1});
        gpu::barrier(session.commands,
            gpu::Stage::compute,
            gpu::Access::shader_write,
            gpu::Stage::compute,
            gpu::Access::shader_read);
        recordSumReduction(
            currentSsim,
            widths[level],
            heights[level],
            layout.ssimSums[level]);
        recordDeviationReduction(
            currentSsim,
            widths[level],
            heights[level],
            static_cast<std::uint32_t>(level),
            layout.ssimSums[level],
            layout.deviationSums[level]);

        if (level + 1u < levelCount) {
            const std::uint64_t nextRgbaBytes = elemCounts[level + 1u] * sizeof(LinearRgba);
            const BufferRegion nextInput1 = ((level & 1u) == 0u)
                                                ? BufferRegion{layout.downsample1.offset, nextRgbaBytes}
                                                : BufferRegion{layout.input1.offset, nextRgbaBytes};
            const BufferRegion nextInput2 = ((level & 1u) == 0u)
                                                ? BufferRegion{layout.downsample2.offset, nextRgbaBytes}
                                                : BufferRegion{layout.input2.offset, nextRgbaBytes};
            const DownsampleParamsData downsampleParams = {
                .inWidth = widths[level],
                .inHeight = heights[level],
                .outWidth = widths[level + 1u],
                .outHeight = heights[level + 1u],
            };
            const auto downsampleWorkgroups = ComputeWorkgroupCounts(
                session, widths[level + 1u], heights[level + 1u]);
            const std::uint32_t downsampleWgX = downsampleWorkgroups[0];
            const std::uint32_t downsampleWgY = downsampleWorkgroups[1];
            gpu::bind_pso(session.commands, session.downsampleShader);
            const std::array<gpu::GpuRange, 2> downsample1 = {{
                DescribeBuffer(resources.workspace, currentInput1),
                DescribeBuffer(resources.workspace, nextInput1),
            }};
            SetBufferAddresses(session, downsample1);
            SetParams(session, downsampleParams);
            gpu::dispatch(session.commands, session.root, {.x = downsampleWgX, .y = downsampleWgY, .z = 1});
            const std::array<gpu::GpuRange, 2> downsample2 = {{
                DescribeBuffer(resources.workspace, currentInput2),
                DescribeBuffer(resources.workspace, nextInput2),
            }};
            SetBufferAddresses(session, downsample2);
            SetParams(session, downsampleParams);
            gpu::dispatch(session.commands, session.root, {.x = downsampleWgX, .y = downsampleWgY, .z = 1});
            gpu::barrier(session.commands,
                gpu::Stage::transfer |
                    gpu::Stage::compute,
                gpu::Access::transfer_read | gpu::Access::shader_write,
                gpu::Stage::compute,
                gpu::Access::shader_read | gpu::Access::shader_write);
        }
    }

    EndTimestamp(session);
    gpu::barrier(session.commands,
        gpu::Stage::compute,
        gpu::Access::shader_write,
        gpu::Stage::transfer,
        gpu::Access::transfer_read);
    std::array<MemoryCopy, kDefaultScaleWeights.size() * 2u> resultCopies{};
    std::uint32_t resultCopyCount = 0;
    for (std::size_t level = 0; level < levelCount; ++level) {
        resultCopies[resultCopyCount++] = {
            .srcOffset = layout.ssimSums[level].offset,
            .dstOffset = sumOutputOffsets[level],
            .size = sizeof(float),
        };
        resultCopies[resultCopyCount++] = {
            .srcOffset = layout.deviationSums[level].offset,
            .dstOffset = deviationOutputOffsets[level],
            .size = sizeof(float),
        };
    }
    CopyRegions(session, resources.workspace, resources.readback,
        resultCopyCount,
        resultCopies.data());
    gpu::barrier(session.commands,
        gpu::Stage::transfer,
        gpu::Access::transfer_write,
        gpu::Stage::host,
        gpu::Access::host_read);
    const std::array<const VulkanVideoFrame*, 2> submittedVideoFrames = {video1, video2};
    SubmitCommands(
        session,
        videoInput ? std::span<const VulkanVideoFrame* const>(submittedVideoFrames) :
                      std::span<const VulkanVideoFrame* const>());
    lock1.Unlock();
    lock2.Unlock();
    outputs.front().dispatchAndSubmit_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - dispatchStartedAt);

    const auto readbackStartedAt = std::chrono::steady_clock::now();
    WaitForSubmission(session);

    const auto* reductionBytes =
        static_cast<const std::uint8_t*>(resources.readback.mapped);
    outputs.front().gpuTimestampMs = ReadGpuTimestampMs(session);
    outputs.front().readback_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - readbackStartedAt);

    const auto postProcessStartedAt = std::chrono::steady_clock::now();
    const auto processLevel = [&](std::size_t level) {
        ScaleOutputs& output = outputs[level];
        output.width = widths[level];
        output.height = heights[level];
        output.elemCount = elemCounts[level];
        float ssimSum = 0.0f;
        float deviationSum = 0.0f;
        std::memcpy(
            &ssimSum,
            reductionBytes + static_cast<std::size_t>(sumOutputOffsets[level]),
            sizeof(ssimSum));
        std::memcpy(
            &deviationSum,
            reductionBytes + static_cast<std::size_t>(deviationOutputOffsets[level]),
            sizeof(deviationSum));
        output.meanSsim =
            static_cast<double>(ssimSum) / static_cast<double>(elemCounts[level]);
        output.ssimScore =
            1.0 - static_cast<double>(deviationSum) /
                      static_cast<double>(elemCounts[level]);
    };
    for (std::size_t level = 0; level < levelCount; ++level) {
        const auto levelStartedAt = std::chrono::steady_clock::now();
        processLevel(level);
        const double levelMs = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - levelStartedAt)
                                   .count();
        if (level == 0u) {
            outputs.front().postProcessBaseScaleMs = levelMs;
        } else {
            outputs.front().postProcessRemainingScalesMs += levelMs;
        }
    }
    outputs.front().postProcess_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - postProcessStartedAt);
    return outputs;
}

DownsampleOutputs RunDownsample2x2Cpu(
    const std::vector<LinearRgba>& input,
    std::uint32_t inWidth,
    std::uint32_t inHeight) {
    const std::size_t inCount = static_cast<std::size_t>(inWidth) * static_cast<std::size_t>(inHeight);
    if (input.size() != inCount) {
        throw std::runtime_error("downsample input size mismatch");
    }
    const std::uint32_t outWidth = inWidth / 2u;
    const std::uint32_t outHeight = inHeight / 2u;
    if (outWidth == 0 || outHeight == 0) {
        throw std::runtime_error("downsample output dimensions are zero");
    }

    DownsampleOutputs out;
    out.width = outWidth;
    out.height = outHeight;
    out.pixels.resize(static_cast<std::size_t>(outWidth) * static_cast<std::size_t>(outHeight));

    ParallelFor(outHeight, [&](std::size_t beginRow, std::size_t endRow) {
        for (std::size_t oy = beginRow; oy < endRow; ++oy) {
            const std::size_t row0 = (oy * 2u) * inWidth;
            const std::size_t row1 = row0 + inWidth;
            for (std::uint32_t ox = 0; ox < outWidth; ++ox) {
                const std::size_t x = static_cast<std::size_t>(ox) * 2u;
                const LinearRgba& p00 = input[row0 + x];
                const LinearRgba& p01 = input[row0 + x + 1u];
                const LinearRgba& p10 = input[row1 + x];
                const LinearRgba& p11 = input[row1 + x + 1u];
                LinearRgba& dst = out.pixels[oy * outWidth + ox];
                dst.r = ((p00.r + p01.r) + p10.r + p11.r) * 0.25f;
                dst.g = ((p00.g + p01.g) + p10.g + p11.g) * 0.25f;
                dst.b = ((p00.b + p01.b) + p10.b + p11.b) * 0.25f;
                dst.a = ((p00.a + p01.a) + p10.a + p11.a) * 0.25f;
            }
        }
    });
    return out;
}

bool HasDeviceExtension(VkPhysicalDevice physicalDevice, const char* extensionName) {
    std::uint32_t count = 0;
    VkCheck(
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr),
        "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> extensions(count);
    VkCheck(
        vkEnumerateDeviceExtensionProperties(
            physicalDevice, nullptr, &count, extensions.data()),
        "vkEnumerateDeviceExtensionProperties");
    return std::any_of(
        extensions.begin(),
        extensions.end(),
        [extensionName](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, extensionName) == 0;
        });
}

struct VideoDeviceConfig {
    GpuSession& session;
    std::vector<VkDeviceQueueCreateInfo> queues;
    float priority = 1.0f;
    VkPhysicalDeviceVideoMaintenance1FeaturesKHR maintenance{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR,
    };
};

VkResult ConfigureVideoDevice(void* context, VkPhysicalDevice physical, VkDeviceCreateInfo& info) {
    auto& config = *static_cast<VideoDeviceConfig*>(context);
    auto& session = config.session;
    config.queues.assign(info.pQueueCreateInfos, info.pQueueCreateInfos + info.queueCreateInfoCount);
    session.videoDeviceExtensions.assign(info.ppEnabledExtensionNames, info.ppEnabledExtensionNames + info.enabledExtensionCount);
    session.physicalDevice = physical;
    session.queueFamilyIndex = config.queues.front().queueFamilyIndex;
    std::uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties2(physical, &queueCount, nullptr);
    std::vector<VkQueueFamilyProperties2> queues(queueCount);
    std::vector<VkQueueFamilyVideoPropertiesKHR> video(queueCount);
    for (std::uint32_t i = 0; i < queueCount; ++i) {
        queues[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        video[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
        queues[i].pNext = &video[i];
    }
    vkGetPhysicalDeviceQueueFamilyProperties2(physical, &queueCount, queues.data());
    const auto& compute = queues[session.queueFamilyIndex].queueFamilyProperties;
    session.computeQueueFlags = compute.queueFlags;
    session.timestampValidBits = compute.timestampValidBits;
    VkVideoCodecOperationFlagsKHR supported = 0;
    if (HasDeviceExtension(physical, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) &&
        HasDeviceExtension(physical, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME)) {
        const std::array<std::pair<const char*, VkVideoCodecOperationFlagsKHR>, 4> codecs{{
            {VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR},
            {VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME, VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR},
            {VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME, VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR},
            {VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME, VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR},
        }};
        for (const auto& [extension, capability] : codecs)
            if (HasDeviceExtension(physical, extension)) {
                supported |= capability;
                session.videoDeviceExtensions.push_back(extension);
            }
        session.videoDeviceExtensions.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
        session.videoDeviceExtensions.push_back(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);
    }
    for (std::uint32_t i = 0; i < queueCount; ++i) {
        const auto& q = queues[i].queueFamilyProperties;
        const auto caps = video[i].videoCodecOperations & supported;
        if (!q.queueCount || !(q.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) || !caps) continue;
        if (session.videoDecodeQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED) {
            session.videoDecodeQueueFamilyIndex = i;
            session.videoDecodeQueueFlags = q.queueFlags;
            session.videoDecodeCaps = caps;
        } else if (session.videoDecodeQueueFamilyIndexSecondary == VK_QUEUE_FAMILY_IGNORED) {
            session.videoDecodeQueueFamilyIndexSecondary = i;
            session.videoDecodeQueueFlagsSecondary = q.queueFlags;
            session.videoDecodeCapsSecondary = caps;
        } else break;
        if (i != session.queueFamilyIndex)
            config.queues.push_back({
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = i, .queueCount = 1, .pQueuePriorities = &config.priority,
            });
    }
    session.videoSupported = session.videoDecodeCaps != 0;
    if (session.videoSupported && HasDeviceExtension(physical, VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME)) {
        VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &config.maintenance};
        vkGetPhysicalDeviceFeatures2(physical, &features);
        if (config.maintenance.videoMaintenance1) {
            session.videoDeviceExtensions.push_back(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);
            config.maintenance.pNext = const_cast<void*>(info.pNext);
            info.pNext = &config.maintenance;
        }
    }
    info.queueCreateInfoCount = static_cast<std::uint32_t>(config.queues.size());
    info.pQueueCreateInfos = config.queues.data();
    info.enabledExtensionCount = static_cast<std::uint32_t>(session.videoDeviceExtensions.size());
    info.ppEnabledExtensionNames = session.videoDeviceExtensions.data();
    return VK_SUCCESS;
}

std::unique_ptr<GpuSession> CreateGpuSession(
    std::span<const std::uint32_t> labPreprocessSpirv,
    std::span<const std::uint32_t> stage0Spirv,
    std::span<const std::uint32_t> stage0ScoreSpirv,
    std::span<const std::uint32_t> reduceSumSpirv,
    std::span<const std::uint32_t> reduceAbsDeviationSpirv,
    std::span<const std::uint32_t> downsampleSpirv,
    std::span<const std::uint32_t> rgba8ToLinearSpirv,
    std::span<const std::uint32_t> vulkanYuvToRgbaSpirv,
    bool enableDebugPipeline,
    bool enableTimestampQueries) {
    auto session = std::make_unique<GpuSession>();

    VideoDeviceConfig videoConfig{.session = *session};
    const gpu::VulkanDeviceConfig interop{.context = &videoConfig, .configure = ConfigureVideoDevice};
    const auto initialized = gpu::create_device({.vulkan = &interop});
    if (!initialized.device) {
        throw std::runtime_error("NoGraphicsAPI device creation failed (Vulkan 1.4, descriptor_heap, "
                                 "device_address_commands, shader_untyped_pointers, mesh_shader "
                                 "and coherent CPU-visible device-local memory are required)");
    }
    session->gpuDevice = initialized.device;
    const auto native = gpu::get_vulkan_device(session->gpuDevice);
    session->instance = native.instance;
    session->physicalDevice = native.physical_device;
    session->device = native.device;
    session->queueFamilyIndex = native.queue_family;
    vkGetPhysicalDeviceProperties(session->physicalDevice, &session->physicalDeviceProperties);
    session->adapterName = gpu::get_device_caps(session->gpuDevice).device_name;
    const auto& limits = session->physicalDeviceProperties.limits;
    if (limits.maxComputeWorkGroupInvocations < 256 || limits.maxComputeWorkGroupSize[0] < 16 ||
        limits.maxComputeWorkGroupSize[1] < 16 || limits.maxComputeSharedMemorySize < 20 * 20 * 16 ||
        gpu::get_device_caps(session->gpuDevice).max_push_data_size < sizeof(ComputeRoot))
        throw std::runtime_error("NoGraphicsAPI device does not meet DSSIM compute limits");
    const auto resourceStartedAt = std::chrono::steady_clock::now();
    session->completion = gpu::create_timeline_semaphore(session->gpuDevice);
    session->timestampQueryEnabled = enableTimestampQueries && session->timestampValidBits != 0;
    if (session->timestampQueryEnabled) {
        const VkQueryPoolCreateInfo queryInfo{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2,
        };
        VkCheck(vkCreateQueryPool(session->device, &queryInfo, nullptr, &session->timestampQueryPool),
                "vkCreateQueryPool");
    }
    session->srgbToLinearLutBuffer = CreateBuffer(
        *session,
        256u * sizeof(float),
        gpu::MemoryType::cpu_visible);
    const auto& lut = SrgbToLinearLut();
    std::memcpy(
        session->srgbToLinearLutBuffer.mapped,
        lut.data(),
        lut.size() * sizeof(float));

    session->initProfiling.createBuffersTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - resourceStartedAt);

    const auto shadersStartedAt = std::chrono::steady_clock::now();
    session->rgba8ToLinearShader =
        gpu::create_compute_pso(session->gpuDevice, rgba8ToLinearSpirv);
    session->preprocessShader =
        gpu::create_compute_pso(session->gpuDevice, labPreprocessSpirv);
    session->stage0ScoreShader =
        gpu::create_compute_pso(session->gpuDevice, stage0ScoreSpirv);
    session->reduceSumShader =
        gpu::create_compute_pso(session->gpuDevice, reduceSumSpirv);
    session->reduceAbsDeviationShader =
        gpu::create_compute_pso(session->gpuDevice, reduceAbsDeviationSpirv);
    if (enableDebugPipeline) {
        session->stage0Shader =
            gpu::create_compute_pso(session->gpuDevice, stage0Spirv);
    }
    session->downsampleShader =
        gpu::create_compute_pso(session->gpuDevice, downsampleSpirv);
    if (session->videoSupported) {
        session->vulkanYuvToRgbaShader = gpu::create_compute_pso(session->gpuDevice, vulkanYuvToRgbaSpirv);
        const auto& caps = gpu::get_device_caps(session->gpuDevice);
        session->videoTextureHeap = gpu::create_gpu_heap(session->gpuDevice, 4 * caps.texture_descriptor_size,
                                                        gpu::MemoryType::texture_descriptor_heap);
        session->videoSamplerHeap = gpu::create_gpu_heap(session->gpuDevice, 2 * caps.sampler_descriptor_size,
                                                        gpu::MemoryType::sampler_descriptor_heap);
        gpu::SamplerDesc sampler{
            .min_filter = gpu::Filter::nearest, .mag_filter = gpu::Filter::nearest,
            .mip_filter = gpu::Filter::nearest,
            .address_u = gpu::AddressMode::clamp_to_edge,
            .address_v = gpu::AddressMode::clamp_to_edge,
            .address_w = gpu::AddressMode::clamp_to_edge,
        };
        gpu::write_sampler_descriptor(session->gpuDevice, session->videoSamplerHeap.range.cpu, sampler);
        sampler.min_filter = gpu::Filter::linear;
        sampler.mag_filter = gpu::Filter::linear;
        gpu::write_sampler_descriptor(session->gpuDevice,
            session->videoSamplerHeap.range.cpu + caps.sampler_descriptor_size, sampler);
        session->writeResourceDescriptors = reinterpret_cast<PFN_vkWriteResourceDescriptorsEXT>(
            vkGetDeviceProcAddr(session->device, "vkWriteResourceDescriptorsEXT"));
        if (!session->writeResourceDescriptors) throw std::runtime_error("missing resource descriptor command");
    }
    session->initProfiling.createShaderModuleTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - shadersStartedAt);
    return session;
}

ProfilingBuckets BuildRuntimeProfilingBuckets(const ProfilingSummary& profiling) {
    return {
        .totalMs = static_cast<double>(profiling.decodeDoneToScoreMs),
        .pipelineSetupMs = static_cast<double>(
            (profiling.createShaderModuleTime + profiling.createPipelineLayoutsTime +
             profiling.createPSOTime)
                .count()),
        .resourcePrepMs = static_cast<double>(
            (profiling.createBuffersTime + profiling.writeInputBuffersTime +
             profiling.createBindGroupsTime)
                .count()),
        .gpuSubmitWaitMs = static_cast<double>(
            (profiling.dispatchAndSubmitTime + profiling.readbackTime).count()),
        .gpuTimestampMs = profiling.gpuTimestampMs,
        .cpuPostProcessMs = static_cast<double>(profiling.postProcessTime.count()),
        .otherMs = static_cast<double>(profiling.otherTime.count()),
    };
}

ProfilingBuckets BuildSessionInitProfilingBuckets(const ProfilingSummary& profiling) {
    const auto pipelineSetupTime =
        profiling.createShaderModuleTime + profiling.createPipelineLayoutsTime + profiling.createPSOTime;
    const auto resourcePrepTime = profiling.createBuffersTime;
    return {
        .totalMs = static_cast<double>((pipelineSetupTime + resourcePrepTime).count()),
        .pipelineSetupMs = static_cast<double>(pipelineSetupTime.count()),
        .resourcePrepMs = static_cast<double>(resourcePrepTime.count()),
        .gpuSubmitWaitMs = 0.0,
        .gpuTimestampMs = 0.0,
        .cpuPostProcessMs = 0.0,
        .otherMs = static_cast<double>(profiling.otherTime.count()),
    };
}

void PrintProfilingBuckets(
    const ProfilingBuckets& buckets,
    const std::string_view prefix,
    const std::string_view label) {
    const auto oldFlags = std::cout.flags();
    const auto oldPrecision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << prefix << label << "total_ms = " << buckets.totalMs << '\n';
    std::cout << prefix << label << "pipeline_setup_ms = " << buckets.pipelineSetupMs << '\n';
    std::cout << prefix << label << "resource_prep_ms = " << buckets.resourcePrepMs << '\n';
    std::cout << prefix << label << "gpu_submit_wait_ms = " << buckets.gpuSubmitWaitMs << '\n';
    std::cout << prefix << label << "gpu_timestamp_ms = " << buckets.gpuTimestampMs << '\n';
    std::cout << prefix << label << "cpu_postprocess_ms = " << buckets.cpuPostProcessMs << '\n';
    std::cout << prefix << label << "other_ms = " << buckets.otherMs << '\n';
    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);
}

RgbaPairComparisonResult CompareRgba8Pair(
    GpuSession& session,
    std::span<const std::uint8_t> rgba1,
    std::span<const std::uint8_t> rgba2,
    std::uint32_t width,
    std::uint32_t height,
    bool collectDebugData,
    const VulkanVideoFrame* video1 = nullptr,
    const VulkanVideoFrame* video2 = nullptr) {
    if (width == 0u || height == 0u) {
        throw std::runtime_error("RGBA8 comparison dimensions must be non-zero");
    }
    const bool videoInput = video1 != nullptr || video2 != nullptr;
    const std::size_t expectedBytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
    if ((!videoInput && (rgba1.size() != expectedBytes || rgba2.size() != expectedBytes)) ||
        (videoInput && (video1 == nullptr || video2 == nullptr))) {
        throw std::runtime_error(
            "RGBA8 comparison buffer size does not match width and height");
    }
    if (videoInput && collectDebugData) {
        throw std::runtime_error("debug dumps are not supported for Vulkan Video frames");
    }
    const auto comparisonStartedAt = std::chrono::steady_clock::now();

    std::vector<LinearRgba> linear1;
    std::vector<LinearRgba> linear2;
    if (collectDebugData) {
        auto input1Future = std::async(
            std::launch::async,
            [rgba1] { return ConvertRgba8ToLinearPlu(rgba1); });
        linear2 = ConvertRgba8ToLinearPlu(rgba2);
        linear1 = input1Future.get();
    }

    RgbaPairComparisonResult result;
    MultiScaleOutputs& compute = result.compute;
    std::vector<std::vector<LinearRgba>> pyramid1;
    std::vector<std::vector<LinearRgba>> pyramid2;
    std::vector<std::uint32_t> pyramidWidths;
    std::vector<std::uint32_t> pyramidHeights;
    pyramidWidths.push_back(width);
    pyramidHeights.push_back(height);

    while (pyramidWidths.size() < kDefaultScaleWeights.size()) {
        const std::uint32_t currWidth = pyramidWidths.back();
        const std::uint32_t currHeight = pyramidHeights.back();
        if (currWidth < 8u || currHeight < 8u) {
            break;
        }
        pyramidWidths.push_back(currWidth / 2u);
        pyramidHeights.push_back(currHeight / 2u);
    }

    milliseconds createShaderModuleProcessingTime{0};
    milliseconds createPSOProcessingTime{0};
    milliseconds createBuffersProcessingTime{0};
    milliseconds writeInputBuffersProcessingTime{0};
    milliseconds createPipelineLayoutsProcessingTime{0};
    milliseconds createBindGroupsProcessingTime{0};
    milliseconds dispatchAndSubmitProcessingTime{0};
    milliseconds readbackProcessingTime{0};
    milliseconds postProcessProcessingTime{0};
    double gpuTimestampProcessingMs = 0.0;
    double postProcessBaseScaleMs = 0.0;
    double postProcessRemainingScalesMs = 0.0;

    if (collectDebugData) {
        pyramid1.push_back(std::move(linear1));
        pyramid2.push_back(std::move(linear2));
        for (std::size_t level = 1; level < pyramidWidths.size(); ++level) {
            const std::uint32_t prevWidth = pyramidWidths[level - 1u];
            const std::uint32_t prevHeight = pyramidHeights[level - 1u];
            auto next1Future = std::async(
                std::launch::async,
                [&pyramid1, prevWidth, prevHeight] {
                    return RunDownsample2x2Cpu(
                        pyramid1.back(),
                        prevWidth,
                        prevHeight);
                });
            DownsampleOutputs next2 =
                RunDownsample2x2Cpu(pyramid2.back(), prevWidth, prevHeight);
            DownsampleOutputs next1 = next1Future.get();
            pyramid1.push_back(std::move(next1.pixels));
            pyramid2.push_back(std::move(next2.pixels));
        }
        for (std::size_t level = 0; level < pyramid1.size(); ++level) {
            ScaleOutputs scale = RunStage0Compute(
                session,
                pyramid1[level],
                pyramid2[level],
                pyramidWidths[level],
                pyramidHeights[level],
                level,
                level == 0);
            compute.scales.push_back(std::move(scale));
        }
    } else {
        compute.scales =
            RunStage0BatchCompute(
                session,
                rgba1,
                rgba2,
                pyramidWidths,
                pyramidHeights,
                video1,
                video2);
    }

    for (const ScaleOutputs& scale : compute.scales) {
        createShaderModuleProcessingTime += scale.createShaderModule_time;
        createPSOProcessingTime += scale.createPSO_time;
        createBuffersProcessingTime += scale.createBuffers_time;
        writeInputBuffersProcessingTime += scale.writeInputBuffers_time;
        createPipelineLayoutsProcessingTime += scale.createPipelineLayouts_time;
        createBindGroupsProcessingTime += scale.createBindGroups_time;
        dispatchAndSubmitProcessingTime += scale.dispatchAndSubmit_time;
        readbackProcessingTime += scale.readback_time;
        postProcessProcessingTime += scale.postProcess_time;
        gpuTimestampProcessingMs += scale.gpuTimestampMs;
        postProcessBaseScaleMs += scale.postProcessBaseScaleMs;
        postProcessRemainingScalesMs += scale.postProcessRemainingScalesMs;
    }

    double weightedSum = 0.0;
    double weightTotal = 0.0;
    for (std::size_t i = 0; i < compute.scales.size(); ++i) {
        const double w = kDefaultScaleWeights[i];
        weightedSum += compute.scales[i].ssimScore * w;
        weightTotal += w;
    }
    compute.weightedSsim = weightedSum / weightTotal;
    compute.score = 1.0 / std::max(compute.weightedSsim, std::numeric_limits<double>::epsilon()) - 1.0;

    if (collectDebugData && pyramid1.size() > 1u && pyramid2.size() > 1u) {
        result.debugScale1Image1 = std::move(pyramid1[1]);
        result.debugScale1Image2 = std::move(pyramid2[1]);
    }

    const auto scoreReadyAt = std::chrono::steady_clock::now();
    const auto comparisonToScoreMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            scoreReadyAt - comparisonStartedAt)
            .count();
    const milliseconds measuredProcessingTime =
        createShaderModuleProcessingTime +
        createPSOProcessingTime +
        createBuffersProcessingTime +
        writeInputBuffersProcessingTime +
        createPipelineLayoutsProcessingTime +
        createBindGroupsProcessingTime +
        dispatchAndSubmitProcessingTime +
        readbackProcessingTime +
        postProcessProcessingTime;
    result.profiling = {
        .decodeDoneToScoreMs = comparisonToScoreMs,
        .createShaderModuleTime = createShaderModuleProcessingTime,
        .createPSOTime = createPSOProcessingTime,
        .createBuffersTime = createBuffersProcessingTime,
        .writeInputBuffersTime = writeInputBuffersProcessingTime,
        .createPipelineLayoutsTime = createPipelineLayoutsProcessingTime,
        .createBindGroupsTime = createBindGroupsProcessingTime,
        .dispatchAndSubmitTime = dispatchAndSubmitProcessingTime,
        .readbackTime = readbackProcessingTime,
        .postProcessTime = postProcessProcessingTime,
        .otherTime = milliseconds(comparisonToScoreMs) - measuredProcessingTime,
        .gpuTimestampMs = gpuTimestampProcessingMs,
        .postProcessBaseScaleMs = postProcessBaseScaleMs,
        .postProcessRemainingScalesMs = postProcessRemainingScalesMs,
    };
    return result;
}

struct OwnedVulkanVideoFrame {
    AVFrame* owner = nullptr;
    VulkanVideoFrame view{};
    std::size_t frameNumber = 0;

    OwnedVulkanVideoFrame() = default;
    OwnedVulkanVideoFrame(const OwnedVulkanVideoFrame&) = delete;
    OwnedVulkanVideoFrame& operator=(const OwnedVulkanVideoFrame&) = delete;
    OwnedVulkanVideoFrame(OwnedVulkanVideoFrame&& other) noexcept
        : owner(std::exchange(other.owner, nullptr)), view(other.view) {
        frameNumber = other.frameNumber;
        if (owner != nullptr) {
            view.frame = owner;
            view.vkFrame = reinterpret_cast<AVVkFrame*>(owner->data[0]);
            view.framesContext = owner->hw_frames_ctx == nullptr
                                     ? nullptr
                                     : reinterpret_cast<AVHWFramesContext*>(owner->hw_frames_ctx->data);
        }
        other.view = {};
        other.frameNumber = 0;
    }
    OwnedVulkanVideoFrame& operator=(OwnedVulkanVideoFrame&& other) noexcept {
        if (this != &other) {
            av_frame_free(&owner);
            owner = std::exchange(other.owner, nullptr);
            view = other.view;
            frameNumber = other.frameNumber;
            if (owner != nullptr) {
                view.frame = owner;
                view.vkFrame = reinterpret_cast<AVVkFrame*>(owner->data[0]);
                view.framesContext = owner->hw_frames_ctx == nullptr
                                         ? nullptr
                                         : reinterpret_cast<AVHWFramesContext*>(owner->hw_frames_ctx->data);
            }
            other.view = {};
            other.frameNumber = 0;
        }
        return *this;
    }
    ~OwnedVulkanVideoFrame() { av_frame_free(&owner); }

    static OwnedVulkanVideoFrame Clone(const VulkanVideoFrame& source) {
        if (source.frame == nullptr) {
            throw std::runtime_error("cannot clone an empty Vulkan Video frame");
        }
        OwnedVulkanVideoFrame result;
        result.owner = av_frame_clone(source.frame);
        if (result.owner == nullptr) {
            throw std::runtime_error("av_frame_clone failed for Vulkan Video frame");
        }
        result.view = source;
        result.view.frame = result.owner;
        result.view.vkFrame = reinterpret_cast<AVVkFrame*>(result.owner->data[0]);
        result.view.framesContext = result.owner->hw_frames_ctx == nullptr
                                        ? nullptr
                                        : reinterpret_cast<AVHWFramesContext*>(result.owner->hw_frames_ctx->data);
        return result;
    }
};

struct VideoDecodePipeline {
    std::mutex mutex;
    std::condition_variable condition;
    std::array<std::deque<OwnedVulkanVideoFrame>, 2> frames;
    std::size_t inFlight = 0;
    std::array<bool, 2> finished = {false, false};
    std::array<bool, 2> decoding = {false, false};
    bool stopRequested = false;
    std::exception_ptr error;
};

void RunVideoComparison(
    GpuSession& session,
    const CliOptions& options,
    const ComparisonRequest& request) {
    if (!IsVideoPath(request.image1.string()) || !IsVideoPath(request.image2.string())) {
        if (options.csvEnabled) {
            throw std::runtime_error("--csv is only supported for video comparisons");
        }
        throw std::runtime_error("video comparison requires two video paths");
    }
    if (!session.videoSupported) {
        throw std::runtime_error(
            "selected Vulkan device does not expose the Vulkan Video decode extensions/queue");
    }

    const std::array<AVCodecID, 2> codecIds = {
        ProbeVideoCodec(request.image1.string()),
        ProbeVideoCodec(request.image2.string()),
    };
    const std::array<VkVideoCodecOperationFlagsKHR, 2> codecOperations = {
        VulkanVideoCodecOperationForCodec(codecIds[0]),
        VulkanVideoCodecOperationForCodec(codecIds[1]),
    };
    if (codecOperations[0] == 0 || codecOperations[1] == 0) {
        throw std::runtime_error(
            "unsupported video codec; expected H.264, HEVC, VP9, or AV1");
    }

    const std::uint32_t primaryQueueFamily = session.videoDecodeQueueFamilyIndex;
    const VkQueueFlags primaryQueueFlags = session.videoDecodeQueueFlags;
    const VkVideoCodecOperationFlagsKHR primaryCaps = session.videoDecodeCaps;
    const bool hasSecondaryQueue =
        session.videoDecodeQueueFamilyIndexSecondary != VK_QUEUE_FAMILY_IGNORED &&
        session.videoDecodeCapsSecondary != 0;
    const std::uint32_t secondaryQueueFamily =
        session.videoDecodeQueueFamilyIndexSecondary;
    const VkQueueFlags secondaryQueueFlags = session.videoDecodeQueueFlagsSecondary;
    const VkVideoCodecOperationFlagsKHR secondaryCaps = session.videoDecodeCapsSecondary;
    const auto primarySupports = [&](std::size_t streamIndex) {
        return (primaryCaps & codecOperations[streamIndex]) == codecOperations[streamIndex];
    };
    const auto secondarySupports = [&](std::size_t streamIndex) {
        return hasSecondaryQueue &&
               (secondaryCaps & codecOperations[streamIndex]) == codecOperations[streamIndex];
    };

    std::array<std::uint32_t, 2> selectedQueueFamilies = {
        primaryQueueFamily,
        primaryQueueFamily,
    };
    std::array<VkQueueFlags, 2> selectedQueueFlags = {
        primaryQueueFlags,
        primaryQueueFlags,
    };
    std::array<VkVideoCodecOperationFlagsKHR, 2> selectedQueueCaps = {
        primaryCaps,
        primaryCaps,
    };
    const bool bothAv1 =
        codecIds[0] == AV_CODEC_ID_AV1 && codecIds[1] == AV_CODEC_ID_AV1;
    if (bothAv1) {
        if (!primarySupports(0) || !primarySupports(1)) {
            throw std::runtime_error("the selected primary Vulkan Video queue does not support both AV1 videos");
        }
    } else if (hasSecondaryQueue) {
        // Prefer putting one stream on the secondary queue while keeping the
        // other stream on the primary queue. This is a capability-based choice;
        // no queue-family index is assumed to have a particular codec set.
        std::size_t secondaryStream = 2;
        if (primarySupports(0) && secondarySupports(1)) {
            secondaryStream = 1;
        } else if (secondarySupports(0) && primarySupports(1)) {
            secondaryStream = 0;
        }
        if (secondaryStream < 2u) {
            selectedQueueFamilies[secondaryStream] = secondaryQueueFamily;
            selectedQueueFlags[secondaryStream] = secondaryQueueFlags;
            selectedQueueCaps[secondaryStream] = secondaryCaps;
        } else if (!primarySupports(0) || !primarySupports(1)) {
            throw std::runtime_error(
                "no compatible Vulkan Video queue-family assignment was found for the two codecs");
        }
    } else if (!primarySupports(0) || !primarySupports(1)) {
        throw std::runtime_error(
            "the available Vulkan Video queue-family does not support both codecs");
    }

    const auto makeVideoInterop = [&](std::size_t streamIndex) {
        return VulkanInteropContext{
            .instance = session.instance,
            .physicalDevice = session.physicalDevice,
            .device = session.device,
            .computeQueueFamily = session.queueFamilyIndex,
            .decodeQueueFamily = selectedQueueFamilies[streamIndex],
            .computeQueueFlags = session.computeQueueFlags,
            .decodeQueueFlags = selectedQueueFlags[streamIndex],
            .decodeVideoCaps = selectedQueueCaps[streamIndex],
            .enabledDeviceExtensions = session.videoDeviceExtensions,
        };
    };
    const auto primaryVideoDevice = VulkanVideoDevice::Create(makeVideoInterop(0));
    const auto secondaryVideoDevice =
        selectedQueueFamilies[0] == selectedQueueFamilies[1]
            ? primaryVideoDevice
            : VulkanVideoDevice::Create(makeVideoInterop(1));
    const std::array<std::shared_ptr<VulkanVideoDevice>, 2> videoDevices = {
        primaryVideoDevice,
        secondaryVideoDevice,
    };
    std::cerr << "[video] codec1=" << avcodec_get_name(codecIds[0])
              << " decode_queue_family1=" << selectedQueueFamilies[0]
              << " codec2=" << avcodec_get_name(codecIds[1])
              << " decode_queue_family2=" << selectedQueueFamilies[1] << '\n';

    double totalScore = 0.0;
    std::size_t frameCount = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    RgbaPairComparisonResult lastComparison;
    ProfilingSummary totalProfiling;
    std::ofstream csvOutput;
    if (options.csvEnabled) {
        const auto csvParent = options.csv.parent_path();
        if (!csvParent.empty()) {
            std::filesystem::create_directories(csvParent);
        }
        csvOutput.open(options.csv, std::ios::out | std::ios::trunc);
        if (!csvOutput) {
            throw std::runtime_error("failed to open video CSV output: " + options.csv.string());
        }
        csvOutput << "time_seconds,frame_number,dssim\n";
        csvOutput << std::fixed << std::setprecision(9);
    }
    const auto videoStartedAt = std::chrono::steady_clock::now();
    double videoTimestampOrigin = std::numeric_limits<double>::quiet_NaN();
    VideoDecodePipeline decodePipeline;
    const std::size_t pipelineDepth = options.pipelineDepth;
    const std::array<std::string, 2> videoPaths = {
        request.image1.string(),
        request.image2.string(),
    };
    const auto decodeStream = [&](std::size_t streamIndex) {
        try {
            {
                std::lock_guard lock(decodePipeline.mutex);
                decodePipeline.decoding[streamIndex] = true;
            }
            decodePipeline.condition.notify_all();
            VulkanVideoReader reader;
            reader.Open(videoPaths[streamIndex], videoDevices[streamIndex]);

            std::size_t frameNumber = 0;
            for (;;) {
                bool reservedPipelineSlot = false;
                {
                    std::unique_lock lock(decodePipeline.mutex);
                    if (streamIndex == 0u) {
                        decodePipeline.condition.wait(lock, [&] {
                            return decodePipeline.stopRequested ||
                                   decodePipeline.inFlight < pipelineDepth;
                        });
                        if (decodePipeline.stopRequested) {
                            break;
                        }
                        ++decodePipeline.inFlight;
                        reservedPipelineSlot = true;
                    } else {
                        decodePipeline.condition.wait(lock, [&] {
                            return decodePipeline.stopRequested ||
                                   decodePipeline.frames[streamIndex].size() < pipelineDepth;
                        });
                        if (decodePipeline.stopRequested) {
                            break;
                        }
                    }
                    decodePipeline.decoding[streamIndex] = true;
                }

                VulkanVideoFrame decoded;
                const bool hasFrame = reader.Next(decoded);
                if (!hasFrame) {
                    std::lock_guard lock(decodePipeline.mutex);
                    decodePipeline.decoding[streamIndex] = false;
                    decodePipeline.finished[streamIndex] = true;
                    if (reservedPipelineSlot) {
                        --decodePipeline.inFlight;
                    }
                    decodePipeline.condition.notify_all();
                    break;
                }

                OwnedVulkanVideoFrame owned = OwnedVulkanVideoFrame::Clone(decoded);
                owned.frameNumber = frameNumber++;
                {
                    std::lock_guard lock(decodePipeline.mutex);
                    if (decodePipeline.stopRequested) {
                        break;
                    }
                    decodePipeline.decoding[streamIndex] = false;
                    decodePipeline.frames[streamIndex].emplace_back(std::move(owned));
                }
                decodePipeline.condition.notify_all();
            }
        } catch (...) {
            std::lock_guard lock(decodePipeline.mutex);
            if (decodePipeline.error == nullptr) {
                decodePipeline.error = std::current_exception();
            }
            decodePipeline.stopRequested = true;
            decodePipeline.decoding[streamIndex] = false;
            decodePipeline.finished[streamIndex] = true;
        }
        {
            std::lock_guard lock(decodePipeline.mutex);
            decodePipeline.decoding[streamIndex] = false;
            decodePipeline.finished[streamIndex] = true;
        }
        decodePipeline.condition.notify_all();
    };

    std::array<std::thread, 2> decodeThreads = {
        std::thread(decodeStream, 0u),
        std::thread(decodeStream, 1u),
    };

    const auto stopDecodeThreads = [&] {
        {
            std::lock_guard lock(decodePipeline.mutex);
            decodePipeline.stopRequested = true;
        }
        decodePipeline.condition.notify_all();
        for (std::thread& thread : decodeThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    };

    try {
        for (;;) {
            OwnedVulkanVideoFrame frame1;
            OwnedVulkanVideoFrame frame2;
            {
                std::unique_lock lock(decodePipeline.mutex);
                decodePipeline.condition.wait(lock, [&] {
                    const bool unmatchedQueuedFrame =
                        (decodePipeline.finished[0] && decodePipeline.frames[0].empty() &&
                         !decodePipeline.frames[1].empty()) ||
                        (decodePipeline.finished[1] && decodePipeline.frames[1].empty() &&
                         !decodePipeline.frames[0].empty());
                    return decodePipeline.error != nullptr ||
                           (!decodePipeline.frames[0].empty() &&
                            !decodePipeline.frames[1].empty()) ||
                           unmatchedQueuedFrame ||
                           (decodePipeline.finished[0] && decodePipeline.finished[1]);
                });
                if (decodePipeline.error != nullptr) {
                    std::rethrow_exception(decodePipeline.error);
                }
                if (decodePipeline.frames[0].empty() || decodePipeline.frames[1].empty()) {
                    if (decodePipeline.finished[0] && decodePipeline.finished[1]) {
                        if (!decodePipeline.frames[0].empty() ||
                            !decodePipeline.frames[1].empty()) {
                            throw std::runtime_error("videos have different decoded frame counts");
                        }
                        break;
                    }
                    if ((decodePipeline.finished[0] && decodePipeline.frames[0].empty() &&
                         !decodePipeline.frames[1].empty()) ||
                        (decodePipeline.finished[1] && decodePipeline.frames[1].empty() &&
                         !decodePipeline.frames[0].empty())) {
                        throw std::runtime_error("videos have different decoded frame counts");
                    }
                    continue;
                }
                frame1 = std::move(decodePipeline.frames[0].front());
                frame2 = std::move(decodePipeline.frames[1].front());
                decodePipeline.frames[0].pop_front();
                decodePipeline.frames[1].pop_front();
            }
            decodePipeline.condition.notify_all();

            if (frame1.frameNumber != frameCount || frame2.frameNumber != frameCount) {
                throw std::runtime_error("video pipeline frame dependency/order was violated");
            }
            if (frame1.view.width != frame2.view.width ||
                frame1.view.height != frame2.view.height) {
                throw std::runtime_error("video frames have different dimensions");
            }
            if (frame1.view.softwareFormat != AV_PIX_FMT_NV12 &&
                frame1.view.softwareFormat != AV_PIX_FMT_P010LE &&
                frame1.view.softwareFormat != AV_PIX_FMT_P010BE) {
                throw std::runtime_error(
                    "unsupported Vulkan Video output format; expected NV12 or P010");
            }
            if (frame2.view.softwareFormat != AV_PIX_FMT_NV12 &&
                frame2.view.softwareFormat != AV_PIX_FMT_P010LE &&
                frame2.view.softwareFormat != AV_PIX_FMT_P010BE) {
                throw std::runtime_error(
                    "unsupported Vulkan Video output format; expected NV12 or P010");
            }
            width = frame1.view.width;
            height = frame1.view.height;
            RgbaPairComparisonResult comparison = CompareRgba8Pair(
                session,
                {},
                {},
                width,
                height,
                false,
                &frame1.view,
                &frame2.view);
            totalScore += comparison.compute.score;
            lastComparison = std::move(comparison);
            totalProfiling.decodeDoneToScoreMs += lastComparison.profiling.decodeDoneToScoreMs;
            totalProfiling.dispatchAndSubmitTime += lastComparison.profiling.dispatchAndSubmitTime;
            totalProfiling.readbackTime += lastComparison.profiling.readbackTime;
            totalProfiling.postProcessTime += lastComparison.profiling.postProcessTime;
            totalProfiling.otherTime += lastComparison.profiling.otherTime;
            totalProfiling.gpuTimestampMs += lastComparison.profiling.gpuTimestampMs;
            const std::size_t frameNumber = frame1.frameNumber;
            ++frameCount;
            if (std::isnan(videoTimestampOrigin)) {
                videoTimestampOrigin = frame1.view.timestampSeconds;
            }
            const double videoTimeSeconds =
                std::max(0.0, frame1.view.timestampSeconds - videoTimestampOrigin);
            const double elapsedSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - videoStartedAt).count();
            const double processingFps = elapsedSeconds > 0.0
                                             ? static_cast<double>(frameCount) / elapsedSeconds
                                             : 0.0;
            const double runningAverage = totalScore / static_cast<double>(frameCount);
            if (csvOutput) {
                csvOutput << videoTimeSeconds << ',' << frameNumber << ','
                          << lastComparison.compute.score << '\n';
            }
            std::size_t currentPipelineDepth = 0;
            {
                std::lock_guard lock(decodePipeline.mutex);
                currentPipelineDepth = decodePipeline.inFlight;
            }
            std::cerr << '\r' << std::fixed << std::setprecision(3)
                      << "[video] fps=" << processingFps
                      << " pipeline_depth=" << currentPipelineDepth
                      << " pipeline_capacity=" << pipelineDepth
                      << " frames=" << frameCount
                      << " elapsed_s=" << elapsedSeconds
                      << " last_frame_dssim=" << std::setprecision(8)
                      << lastComparison.compute.score
                      << " average_dssim=" << runningAverage << std::flush;
            {
                std::lock_guard lock(decodePipeline.mutex);
                --decodePipeline.inFlight;
            }
            decodePipeline.condition.notify_all();
        }
        stopDecodeThreads();
    } catch (...) {
        stopDecodeThreads();
        throw;
    }
    if (frameCount == 0) {
        throw std::runtime_error("no decodable Vulkan Video frames were found");
    }
    std::cerr << '\n';

    const double averageScore = totalScore / static_cast<double>(frameCount);
    lastComparison.compute.score = averageScore;
    lastComparison.compute.weightedSsim = 1.0 / (1.0 + averageScore);
    const DecodedInputInfo decoded1 = {
        .width = width,
        .height = height,
        .channels = 4,
        .byteCount = static_cast<std::size_t>(width) * height * 4u,
    };
    const DecodedInputInfo decoded2 = decoded1;
    CliOptions resultOptions = options;
    resultOptions.image1 = request.image1;
    resultOptions.image2 = request.image2;
    if (!resultOptions.out.empty()) {
        WriteStringFile(
            resultOptions.out,
            BuildJson(
                resultOptions,
                session.adapterName,
                decoded1,
                decoded2,
                lastComparison.compute,
                totalProfiling,
                nullptr));
    }
    std::cout << std::fixed << std::setprecision(8) << averageScore << '\t'
              << resultOptions.image2.string() << "\tframes=" << frameCount << '\n';
    if (options.profilingEnabled) {
        PrintProfilingBuckets(
            BuildRuntimeProfilingBuckets(totalProfiling),
            "[profiling] ",
            "video_");
    }
}

void RunComparison(
    GpuSession& session,
    const CliOptions& options,
    const ComparisonRequest& request) {
    if (IsVideoPath(request.image1.string()) || IsVideoPath(request.image2.string())) {
        RunVideoComparison(session, options, request);
        return;
    }
    if (options.csvEnabled) {
        throw std::runtime_error("--csv is only supported for video comparisons");
    }
    const DecodedImage image1 = LoadImageRgba8(request.image1);
    const DecodedImage image2 = LoadImageRgba8(request.image2);
    if (image1.pixels.empty() || image2.pixels.empty()) {
        throw std::runtime_error("decoded image pixels are empty");
    }
    if (image1.width != image2.width || image1.height != image2.height) {
        throw std::runtime_error(
            "image size mismatch; multi-scale stage requires identical dimensions");
    }

    RgbaPairComparisonResult comparison = CompareRgba8Pair(
        session,
        image1.pixels,
        image2.pixels,
        image1.width,
        image1.height,
        options.debugDumpEnabled);
    MultiScaleOutputs& compute = comparison.compute;

    const DecodedInputInfo decoded1 = {
        .width = image1.width,
        .height = image1.height,
        .channels = image1.channels,
        .byteCount = image1.pixels.size(),
    };
    const DecodedInputInfo decoded2 = {
        .width = image2.width,
        .height = image2.height,
        .channels = image2.channels,
        .byteCount = image2.pixels.size(),
    };

    DebugDumpInfo debugInfo;
    DebugDumpInfo* debugInfoPtr = nullptr;
    if (options.debugDumpEnabled) {
        std::filesystem::create_directories(options.debugDumpDir);
        debugInfo.image1RgbaPath = options.debugDumpDir / "image1_rgba8.gpu.bin";
        debugInfo.image2RgbaPath = options.debugDumpDir / "image2_rgba8.gpu.bin";
        debugInfo.stage0DssimPath = options.debugDumpDir / "stage0_dssim5x5_gaussian_linear_u32le.gpu.bin";
        debugInfo.stage0Mu1Path = options.debugDumpDir / "stage0_mu1_f32le.gpu.bin";
        debugInfo.stage0Mu2Path = options.debugDumpDir / "stage0_mu2_f32le.gpu.bin";
        debugInfo.stage0Var1Path = options.debugDumpDir / "stage0_var1_f32le.gpu.bin";
        debugInfo.stage0Var2Path = options.debugDumpDir / "stage0_var2_f32le.gpu.bin";
        debugInfo.stage0Cov12Path = options.debugDumpDir / "stage0_cov12_f32le.gpu.bin";
        debugInfo.stage0ElemCount =
            compute.scales.empty() ? 0 : compute.scales[0].elemCount;
        WriteU8Buffer(debugInfo.image1RgbaPath, image1.pixels);
        WriteU8Buffer(debugInfo.image2RgbaPath, image2.pixels);
        WriteF32LeBuffer(debugInfo.stage0DssimPath, compute.scales[0].ssimMap);
        WriteF32LeBuffer(debugInfo.stage0Mu1Path, compute.scales[0].mu1);
        WriteF32LeBuffer(debugInfo.stage0Mu2Path, compute.scales[0].mu2);
        WriteF32LeBuffer(debugInfo.stage0Var1Path, compute.scales[0].var1);
        WriteF32LeBuffer(debugInfo.stage0Var2Path, compute.scales[0].var2);
        WriteF32LeBuffer(debugInfo.stage0Cov12Path, compute.scales[0].cov12);
        if (compute.scales.size() > 1u &&
            !comparison.debugScale1Image1.empty() &&
            !comparison.debugScale1Image2.empty()) {
            debugInfo.image1Scale1Path = options.debugDumpDir / "image1_scale1_rgba8.gpu.bin";
            debugInfo.image2Scale1Path = options.debugDumpDir / "image2_scale1_rgba8.gpu.bin";
            debugInfo.stage1DssimPath = options.debugDumpDir / "stage1_dssim5x5_gaussian_linear_u32le.gpu.bin";
            debugInfo.stage1ElemCount = compute.scales[1].elemCount;
            WriteU8Buffer(
                debugInfo.image1Scale1Path,
                ConvertLinearPluToRgba8(comparison.debugScale1Image1));
            WriteU8Buffer(
                debugInfo.image2Scale1Path,
                ConvertLinearPluToRgba8(comparison.debugScale1Image2));
            WriteF32LeBuffer(
                debugInfo.stage1DssimPath,
                compute.scales[1].ssimMap);
        }
        debugInfoPtr = &debugInfo;
    }

    std::ostringstream scoreText;
    scoreText << std::fixed << std::setprecision(8) << compute.score;
    CliOptions resultOptions = options;
    resultOptions.image1 = request.image1;
    resultOptions.image2 = request.image2;

    if (!resultOptions.out.empty()) {
        const std::string json =
            BuildJson(
                resultOptions,
                session.adapterName,
                decoded1,
                decoded2,
                compute,
                comparison.profiling,
                debugInfoPtr);
        WriteStringFile(resultOptions.out, json);
    }

    std::cout << scoreText.str() << '\t' << resultOptions.image2.string() << '\n';
    if (resultOptions.profilingEnabled) {
        PrintProfilingBuckets(
            BuildRuntimeProfilingBuckets(comparison.profiling),
            "[profiling] ",
            "");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = ParseArgs(argc, argv);
        const auto stage0ShaderPath = ResolveShaderPath(argv[0], "stage0_absdiff.spv");
        const auto stage0ScoreShaderPath = ResolveShaderPath(argv[0], "stage0_score.spv");
        const auto reduceSumShaderPath = ResolveShaderPath(argv[0], "reduce_sum.spv");
        const auto reduceAbsDeviationShaderPath =
            ResolveShaderPath(argv[0], "reduce_abs_deviation.spv");
        const auto labPreprocessShaderPath = ResolveShaderPath(argv[0], "lab_preprocess.spv");
        const auto downsampleShaderPath = ResolveShaderPath(argv[0], "downsample_2x2.spv");
        const auto rgba8ToLinearShaderPath = ResolveShaderPath(argv[0], "rgba8_to_linear.spv");
        const auto vulkanYuvToRgbaShaderPath = ResolveShaderPath(argv[0], "vulkan_yuv_to_rgba8.spv");
        const auto stage0Spirv = ReadSpirv(stage0ShaderPath);
        const auto stage0ScoreSpirv = ReadSpirv(stage0ScoreShaderPath);
        const auto reduceSumSpirv = ReadSpirv(reduceSumShaderPath);
        const auto reduceAbsDeviationSpirv = ReadSpirv(reduceAbsDeviationShaderPath);
        const auto labPreprocessSpirv = ReadSpirv(labPreprocessShaderPath);
        const auto downsampleSpirv = ReadSpirv(downsampleShaderPath);
        const auto rgba8ToLinearSpirv = ReadSpirv(rgba8ToLinearShaderPath);
        const auto vulkanYuvToRgbaSpirv = ReadSpirv(vulkanYuvToRgbaShaderPath);
        auto session =
            CreateGpuSession(
                labPreprocessSpirv,
                stage0Spirv,
                stage0ScoreSpirv,
                reduceSumSpirv,
                reduceAbsDeviationSpirv,
                downsampleSpirv,
                rgba8ToLinearSpirv,
                vulkanYuvToRgbaSpirv,
                options.debugDumpEnabled,
                options.profilingEnabled || !options.out.empty());
        if (options.profilingEnabled) {
            PrintProfilingBuckets(
                BuildSessionInitProfilingBuckets(session->initProfiling),
                "[profiling] ",
                "session_init_");
        }

        if (options.stdinPairsMode) {
            std::string line;
            std::size_t lineNumber = 0;
            while (std::getline(std::cin, line)) {
                ++lineNumber;
                if (line.empty()) {
                    continue;
                }
                ComparisonRequest request;
                try {
                    request = ParseComparisonRequestLine(line);
                } catch (const std::exception& ex) {
                    throw std::runtime_error(
                        "failed to parse stdin pair at line " + std::to_string(lineNumber) + ": " + ex.what());
                }
                try {
                    RunComparison(*session, options, request);
                } catch (const std::exception& ex) {
                    throw std::runtime_error(
                        "comparison failed at line " + std::to_string(lineNumber) + ": " + ex.what());
                }
            }
        } else {
            RunComparison(
                *session,
                options,
                ComparisonRequest{
                    .image1 = options.image1,
                    .image2 = options.image2,
                });
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "dssim-Vulkan error: " << ex.what() << '\n';
        return 1;
    }
}

