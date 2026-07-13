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

#include <vulkan/vulkan.h>

#include "png_loader.h"
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

struct CliOptions {
    std::filesystem::path image1;
    std::filesystem::path image2;
    std::filesystem::path out;
    std::filesystem::path csv;
    std::filesystem::path debugDumpDir;
    bool debugDumpEnabled = false;
    bool stdinPairsMode = false;
    bool profilingEnabled = false;
    bool csvEnabled = false;
    std::size_t pipelineDepth = 3;
    bool pipelineDepthExplicit = false;
};

struct ComparisonRequest {
    std::filesystem::path image1;
    std::filesystem::path image2;
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

struct VulkanBuffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    bool hostCoherent = false;

    VulkanBuffer() = default;
    ~VulkanBuffer() { Reset(); }
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    VulkanBuffer(VulkanBuffer&& other) noexcept { *this = std::move(other); }
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept {
        if (this != &other) {
            Reset();
            device = std::exchange(other.device, VK_NULL_HANDLE);
            buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
            memory = std::exchange(other.memory, VK_NULL_HANDLE);
            size = std::exchange(other.size, 0);
            mapped = std::exchange(other.mapped, nullptr);
            hostCoherent = std::exchange(other.hostCoherent, false);
        }
        return *this;
    }
    explicit operator bool() const noexcept { return buffer != VK_NULL_HANDLE; }
    void Reset() noexcept {
        if (device != VK_NULL_HANDLE) {
            if (mapped != nullptr && memory != VK_NULL_HANDLE) {
                vkUnmapMemory(device, memory);
            }
            if (buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, buffer, nullptr);
            }
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, memory, nullptr);
            }
        }
        device = VK_NULL_HANDLE;
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        size = 0;
        mapped = nullptr;
        hostCoherent = false;
    }
};

struct Stage0Resources {
    VkDeviceSize workspaceCapacity = 0;
    VkDeviceSize uploadCapacity = 0;
    VkDeviceSize readbackCapacity = 0;
    VulkanBuffer workspace;
    VulkanBuffer upload;
    VulkanBuffer readback;
};

struct BatchStage0Resources {
    VkDeviceSize workspaceCapacity = 0;
    VkDeviceSize uploadCapacity = 0;
    VkDeviceSize readbackCapacity = 0;
    VulkanBuffer workspace;
    VulkanBuffer upload;
    VulkanBuffer readback;
};

struct ComputeShader {
    VkShaderEXT shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
};

struct GpuSession {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
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
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence submitFence = VK_NULL_HANDLE;

    PFN_vkCreateShadersEXT createShaders = nullptr;
    PFN_vkDestroyShaderEXT destroyShader = nullptr;
    PFN_vkCmdBindShadersEXT cmdBindShaders = nullptr;
    PFN_vkCmdPushDescriptorSetKHR cmdPushDescriptorSet = nullptr;

    ComputeShader preprocessShader;
    ComputeShader stage0Shader;
    ComputeShader stage0ScoreShader;
    ComputeShader downsampleShader;
    ComputeShader rgba8ToLinearShader;
    ComputeShader vulkanYuvToRgbaShader;
    VkSampler videoYSampler = VK_NULL_HANDLE;
    VkSampler videoUvSampler = VK_NULL_HANDLE;
    VulkanBuffer srgbToLinearLutBuffer;

    std::unique_ptr<Stage0Resources> debugStage0Resources;
    std::unique_ptr<BatchStage0Resources> batchStage0Resources;

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

CliOptions ParseArgs(int argc, char** argv) {
    CliOptions options;
    int positionalCount = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--stdin-pairs") {
            options.stdinPairsMode = true;
            continue;
        }
        if (arg == "--profiling") {
            options.profilingEnabled = true;
            continue;
        }

        if (arg == "--pipeline-depth") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --pipeline-depth");
            }
            const std::string value = argv[++i];
            try {
                std::size_t parsedChars = 0;
                const unsigned long long parsed = std::stoull(value, &parsedChars);
                if (parsedChars != value.size() || parsed == 0u ||
                    parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
                    throw std::runtime_error("invalid value");
                }
                options.pipelineDepth = static_cast<std::size_t>(parsed);
                options.pipelineDepthExplicit = true;
            } catch (const std::exception&) {
                throw std::runtime_error("--pipeline-depth must be a positive integer");
            }
            continue;
        }
        if (arg.rfind("--pipeline-depth=", 0) == 0) {
            const std::string value = arg.substr(std::string("--pipeline-depth=").size());
            try {
                std::size_t parsedChars = 0;
                const unsigned long long parsed = std::stoull(value, &parsedChars);
                if (parsedChars != value.size() || parsed == 0u ||
                    parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
                    throw std::runtime_error("invalid value");
                }
                options.pipelineDepth = static_cast<std::size_t>(parsed);
                options.pipelineDepthExplicit = true;
            } catch (const std::exception&) {
                throw std::runtime_error("--pipeline-depth must be a positive integer");
            }
            continue;
        }

        if (arg == "--out") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --out");
            }
            options.out = argv[++i];
            continue;
        }
        if (arg.rfind("--out=", 0) == 0) {
            options.out = arg.substr(std::string("--out=").size());
            continue;
        }

        if (arg == "--csv") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --csv");
            }
            options.csv = argv[++i];
            options.csvEnabled = true;
            continue;
        }
        if (arg.rfind("--csv=", 0) == 0) {
            options.csv = arg.substr(std::string("--csv=").size());
            options.csvEnabled = true;
            continue;
        }

        if (arg == "--debug-dump-dir") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --debug-dump-dir");
            }
            options.debugDumpDir = argv[++i];
            options.debugDumpEnabled = true;
            continue;
        }
        if (arg.rfind("--debug-dump-dir=", 0) == 0) {
            options.debugDumpDir = arg.substr(std::string("--debug-dump-dir=").size());
            options.debugDumpEnabled = true;
            continue;
        }

        if (!arg.empty() && arg[0] != '-') {
            if (positionalCount == 0) {
                options.image1 = arg;
            } else if (positionalCount == 1) {
                options.image2 = arg;
            } else {
                throw std::runtime_error("too many positional arguments");
            }
            ++positionalCount;
            continue;
        }

        throw std::runtime_error("unknown argument: " + arg);
    }

    if (options.debugDumpEnabled && options.debugDumpDir.empty()) {
        throw std::runtime_error("empty --debug-dump-dir");
    }
    if (options.csvEnabled && options.csv.empty()) {
        throw std::runtime_error("empty --csv path");
    }
    if (options.stdinPairsMode) {
        if (positionalCount != 0) {
            throw std::runtime_error("--stdin-pairs does not accept positional image arguments");
        }
        if (!options.out.empty()) {
            throw std::runtime_error("--stdin-pairs cannot be combined with --out");
        }
        if (options.csvEnabled) {
            throw std::runtime_error("--stdin-pairs cannot be combined with --csv");
        }
        if (options.debugDumpEnabled) {
            throw std::runtime_error("--stdin-pairs cannot be combined with --debug-dump-dir");
        }
        if (options.pipelineDepthExplicit) {
            throw std::runtime_error("--stdin-pairs cannot be combined with --pipeline-depth");
        }
    } else if (positionalCount != 2) {
        throw std::runtime_error(
            "usage: dssim-WebGPU <img1> <img2> [--out <json>] "
            "[--csv <path>] [--pipeline-depth <N>] [--debug-dump-dir <dir>] "
            "[--stdin-pairs] [--profiling]");
    }

    return options;
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
    command << "dssim-WebGPU \"" << abs1 << "\" \"" << abs2 << "\"";
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

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
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

std::uint32_t FindMemoryType(
    const GpuSession& session,
    std::uint32_t allowedTypes,
    VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(session.physicalDevice, &properties);
    std::uint32_t bestIndex = std::numeric_limits<std::uint32_t>::max();
    int bestScore = -1;
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((allowedTypes & (1u << i)) == 0u) {
            continue;
        }
        const VkMemoryPropertyFlags flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) {
            continue;
        }
        int score = 0;
        for (std::uint32_t bit = 0; bit < 32u; ++bit) {
            score += ((flags & preferred & (1u << bit)) != 0u) ? 1 : 0;
        }
        if (score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }
    if (bestIndex == std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("no compatible Vulkan memory type");
    }
    return bestIndex;
}

VulkanBuffer CreateBuffer(
    const GpuSession& session,
    VkDeviceSize byteSize,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred = 0) {
    VulkanBuffer result;
    result.device = session.device;
    result.size = std::max<VkDeviceSize>(byteSize, 4u);
    const VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = result.size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkCheck(vkCreateBuffer(session.device, &bufferInfo, nullptr, &result.buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(session.device, result.buffer, &requirements);
    const std::uint32_t memoryTypeIndex =
        FindMemoryType(session, requirements.memoryTypeBits, required, preferred);
    const VkMemoryAllocateInfo allocationInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = memoryTypeIndex,
    };
    VkCheck(
        vkAllocateMemory(session.device, &allocationInfo, nullptr, &result.memory),
        "vkAllocateMemory");
    VkCheck(vkBindBufferMemory(session.device, result.buffer, result.memory, 0), "vkBindBufferMemory");

    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(session.physicalDevice, &memoryProperties);
    const VkMemoryPropertyFlags actualFlags =
        memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags;
    result.hostCoherent = (actualFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    if ((actualFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        VkCheck(
            vkMapMemory(session.device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped),
            "vkMapMemory");
    }
    return result;
}

void FlushMappedBuffer(const VulkanBuffer& buffer) {
    if (buffer.mapped == nullptr) {
        throw std::runtime_error("attempted to flush an unmapped Vulkan buffer");
    }
    if (!buffer.hostCoherent) {
        const VkMappedMemoryRange range = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = buffer.memory,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        VkCheck(vkFlushMappedMemoryRanges(buffer.device, 1, &range), "vkFlushMappedMemoryRanges");
    }
}

void InvalidateMappedBuffer(const VulkanBuffer& buffer) {
    if (buffer.mapped == nullptr) {
        throw std::runtime_error("attempted to invalidate an unmapped Vulkan buffer");
    }
    if (!buffer.hostCoherent) {
        const VkMappedMemoryRange range = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = buffer.memory,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        VkCheck(
            vkInvalidateMappedMemoryRanges(buffer.device, 1, &range),
            "vkInvalidateMappedMemoryRanges");
    }
}

void DestroyComputeShader(GpuSession& session, ComputeShader& shader) noexcept {
    if (shader.shader != VK_NULL_HANDLE && session.destroyShader != nullptr) {
        session.destroyShader(session.device, shader.shader, nullptr);
    }
    if (shader.pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(session.device, shader.pipelineLayout, nullptr);
    }
    if (shader.descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(session.device, shader.descriptorSetLayout, nullptr);
    }
    shader = {};
}

GpuSession::~GpuSession() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        debugStage0Resources.reset();
        batchStage0Resources.reset();
        srgbToLinearLutBuffer.Reset();
        DestroyComputeShader(*this, rgba8ToLinearShader);
        DestroyComputeShader(*this, preprocessShader);
        DestroyComputeShader(*this, stage0Shader);
        DestroyComputeShader(*this, stage0ScoreShader);
        DestroyComputeShader(*this, downsampleShader);
        DestroyComputeShader(*this, vulkanYuvToRgbaShader);
        if (videoYSampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, videoYSampler, nullptr);
        }
        if (videoUvSampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, videoUvSampler, nullptr);
        }
        if (timestampQueryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, timestampQueryPool, nullptr);
        }
        if (submitFence != VK_NULL_HANDLE) {
            vkDestroyFence(device, submitFence, nullptr);
        }
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
        }
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

void BeginCommands(GpuSession& session) {
    VkCheck(vkResetCommandBuffer(session.commandBuffer, 0), "vkResetCommandBuffer");
    const VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VkCheck(vkBeginCommandBuffer(session.commandBuffer, &beginInfo), "vkBeginCommandBuffer");
    if (session.timestampQueryEnabled) {
        vkCmdResetQueryPool(session.commandBuffer, session.timestampQueryPool, 0, 2);
    }
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
    VkCheck(vkEndCommandBuffer(session.commandBuffer), "vkEndCommandBuffer");
    VkCheck(vkResetFences(session.device, 1, &session.submitFence), "vkResetFences");

    std::array<VkSemaphore, 2> waitSemaphores{};
    std::array<VkSemaphore, 2> signalSemaphores{};
    std::array<std::uint64_t, 2> waitValues{};
    std::array<std::uint64_t, 2> signalValues{};
    std::array<VkPipelineStageFlags, 2> waitStages{};
    std::uint32_t semaphoreCount = 0;
    for (const VulkanVideoFrame* frame : videoFrames) {
        if (frame == nullptr || frame->vkFrame == nullptr || frame->vkFrame->sem[0] == VK_NULL_HANDLE) {
            continue;
        }
        const VkSemaphore semaphore = frame->vkFrame->sem[0];
        bool alreadyAdded = false;
        for (std::uint32_t i = 0; i < semaphoreCount; ++i) {
            alreadyAdded = alreadyAdded || waitSemaphores[i] == semaphore;
        }
        if (!alreadyAdded) {
            if (semaphoreCount == waitSemaphores.size()) {
                throw std::runtime_error("too many Vulkan Video synchronization semaphores");
            }
            waitSemaphores[semaphoreCount] = semaphore;
            signalSemaphores[semaphoreCount] = semaphore;
            waitValues[semaphoreCount] = frame->vkFrame->sem_value[0];
            signalValues[semaphoreCount] = waitValues[semaphoreCount] + 1u;
            waitStages[semaphoreCount] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            ++semaphoreCount;
        }
    }
    const VkTimelineSemaphoreSubmitInfo timelineInfo = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = semaphoreCount,
        .pWaitSemaphoreValues = waitValues.data(),
        .signalSemaphoreValueCount = semaphoreCount,
        .pSignalSemaphoreValues = signalValues.data(),
    };
    const VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = semaphoreCount > 0 ? &timelineInfo : nullptr,
        .waitSemaphoreCount = semaphoreCount,
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = waitStages.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &session.commandBuffer,
        .signalSemaphoreCount = semaphoreCount,
        .pSignalSemaphores = signalSemaphores.data(),
    };
    VkCheck(vkQueueSubmit(session.queue, 1, &submitInfo, session.submitFence), "vkQueueSubmit");

    for (const VulkanVideoFrame* frame : videoFrames) {
        if (frame != nullptr && frame->vkFrame != nullptr && frame->vkFrame->sem[0] != VK_NULL_HANDLE) {
            bool alreadyUpdated = false;
            for (const VulkanVideoFrame* previous : videoFrames) {
                if (previous == frame) {
                    break;
                }
                if (previous != nullptr && previous->vkFrame != nullptr &&
                    previous->vkFrame->sem[0] == frame->vkFrame->sem[0]) {
                    alreadyUpdated = true;
                    break;
                }
            }
            if (!alreadyUpdated) {
                frame->vkFrame->sem_value[0]++;
            }
        }
    }
}

void WaitForSubmission(GpuSession& session) {
    VkCheck(
        vkWaitForFences(session.device, 1, &session.submitFence, VK_TRUE, UINT64_MAX),
        "vkWaitForFences");
}

void MemoryBarrier(
    VkCommandBuffer commandBuffer,
    VkPipelineStageFlags2 srcStage,
    VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage,
    VkAccessFlags2 dstAccess) {
    const VkMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
    };
    const VkDependencyInfo dependency = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void BindShader(GpuSession& session, const ComputeShader& shader) {
    const VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
    session.cmdBindShaders(session.commandBuffer, 1, &stage, &shader.shader);
}

void PushStorageDescriptors(
    GpuSession& session,
    const ComputeShader& shader,
    std::span<const VkDescriptorBufferInfo> bufferInfos,
    std::uint32_t firstBinding = 0) {
    if (bufferInfos.size() > 8u) {
        throw std::runtime_error("too many push descriptors");
    }
    std::array<VkWriteDescriptorSet, 8> writes{};
    for (std::size_t i = 0; i < bufferInfos.size(); ++i) {
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = firstBinding + static_cast<std::uint32_t>(i),
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bufferInfos[i],
        };
    }
    session.cmdPushDescriptorSet(
        session.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader.pipelineLayout,
        0,
        static_cast<std::uint32_t>(bufferInfos.size()),
        writes.data());
}

void PushSampledImageDescriptors(
    GpuSession& session,
    const ComputeShader& shader,
    std::span<const VkDescriptorImageInfo> imageInfos,
    std::uint32_t firstBinding = 0) {
    if (imageInfos.size() > 8u) {
        throw std::runtime_error("too many image descriptors");
    }
    std::array<VkWriteDescriptorSet, 8> writes{};
    for (std::size_t i = 0; i < imageInfos.size(); ++i) {
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = firstBinding + static_cast<std::uint32_t>(i),
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &imageInfos[i],
        };
    }
    session.cmdPushDescriptorSet(
        session.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader.pipelineLayout,
        0,
        static_cast<std::uint32_t>(imageInfos.size()),
        writes.data());
}

template <typename Params>
void PushParams(GpuSession& session, const ComputeShader& shader, const Params& params) {
    static_assert(sizeof(Params) == 4u * sizeof(std::uint32_t));
    vkCmdPushConstants(
        session.commandBuffer,
        shader.pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(Params),
        &params);
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
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
};

struct ArenaBuilder {
    VkDeviceSize cursor = 0;
    VkDeviceSize alignment = 4;

    BufferRegion Add(VkDeviceSize byteSize) {
        cursor = AlignUp(cursor, alignment);
        const BufferRegion region = {.offset = cursor, .size = byteSize};
        cursor += byteSize;
        return region;
    }
};

VkDescriptorBufferInfo DescribeBuffer(
    const VulkanBuffer& buffer,
    const BufferRegion& region) {
    return {
        .buffer = buffer.buffer,
        .offset = region.offset,
        .range = region.size,
    };
}

void ValidateStorageRange(
    const GpuSession& session,
    VkDeviceSize byteSize,
    const std::string_view label) {
    if (byteSize > session.physicalDeviceProperties.limits.maxStorageBufferRange) {
        throw std::runtime_error(
            std::string(label) + " exceeds maxStorageBufferRange");
    }
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
    VkDeviceSize workspaceBytes = 0;
    VkDeviceSize uploadBytes = 0;
    VkDeviceSize readbackBytes = 0;
};

DebugStageLayout BuildDebugStageLayout(
    const GpuSession& session,
    std::size_t elemCount,
    bool includeStats) {
    const VkDeviceSize rgbaBytes = elemCount * sizeof(LinearRgba);
    const VkDeviceSize f32Bytes = elemCount * sizeof(float);
    ValidateStorageRange(session, rgbaBytes, "debug RGBA/LAB buffer");
    ValidateStorageRange(session, f32Bytes, "debug SSIM/statistics buffer");
    ArenaBuilder workspace{
        .alignment = std::max<VkDeviceSize>(
            4u,
            session.physicalDeviceProperties.limits.minStorageBufferOffsetAlignment),
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
    const VkDeviceSize rgbaBytes = elemCount * sizeof(LinearRgba);
    const VkDeviceSize f32Bytes = elemCount * sizeof(float);

    ScaleOutputs outputs;
    if (!session.debugStage0Resources ||
        session.debugStage0Resources->workspaceCapacity < layout.workspaceBytes ||
        session.debugStage0Resources->uploadCapacity < layout.uploadBytes ||
        session.debugStage0Resources->readbackCapacity < layout.readbackBytes) {
        const auto startedAt = std::chrono::steady_clock::now();
        auto resources = std::make_unique<Stage0Resources>();
        resources->workspaceCapacity = layout.workspaceBytes;
        resources->uploadCapacity = layout.uploadBytes;
        resources->readbackCapacity = layout.readbackBytes;
        resources->workspace = CreateBuffer(
            session,
            layout.workspaceBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        resources->upload = CreateBuffer(
            session,
            layout.uploadBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        resources->readback = CreateBuffer(
            session,
            layout.readbackBytes,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        session.debugStage0Resources = std::move(resources);
        outputs.createBuffers_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startedAt);
    }
    Stage0Resources& resources = *session.debugStage0Resources;

    const auto writeStartedAt = std::chrono::steady_clock::now();
    auto* uploadBytes = static_cast<std::uint8_t*>(resources.upload.mapped);
    std::memcpy(uploadBytes + layout.upload1.offset, input1.data(), rgbaBytes);
    std::memcpy(uploadBytes + layout.upload2.offset, input2.data(), rgbaBytes);
    FlushMappedBuffer(resources.upload);
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
    const std::array<VkBufferCopy, 2> uploads = {{
        {.srcOffset = layout.upload1.offset, .dstOffset = layout.input1.offset, .size = rgbaBytes},
        {.srcOffset = layout.upload2.offset, .dstOffset = layout.input2.offset, .size = rgbaBytes},
    }};
    vkCmdCopyBuffer(
        session.commandBuffer,
        resources.upload.buffer,
        resources.workspace.buffer,
        static_cast<std::uint32_t>(uploads.size()),
        uploads.data());
    MemoryBarrier(
        session.commandBuffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    BeginTimestamp(session);

    BindShader(session, session.preprocessShader);
    const std::array<VkDescriptorBufferInfo, 2> preprocess1 = {{
        DescribeBuffer(resources.workspace, layout.input1),
        DescribeBuffer(resources.workspace, layout.lab1),
    }};
    PushStorageDescriptors(session, session.preprocessShader, preprocess1);
    PushParams(session, session.preprocessShader, params);
    vkCmdDispatch(session.commandBuffer, wgX, wgY, 1);
    const std::array<VkDescriptorBufferInfo, 2> preprocess2 = {{
        DescribeBuffer(resources.workspace, layout.input2),
        DescribeBuffer(resources.workspace, layout.lab2),
    }};
    PushStorageDescriptors(session, session.preprocessShader, preprocess2);
    PushParams(session, session.preprocessShader, params);
    vkCmdDispatch(session.commandBuffer, wgX, wgY, 1);
    MemoryBarrier(
        session.commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    if (readIntermediateStats) {
        BindShader(session, session.stage0Shader);
        const std::array<VkDescriptorBufferInfo, 8> descriptors = {{
            DescribeBuffer(resources.workspace, layout.lab1),
            DescribeBuffer(resources.workspace, layout.lab2),
            DescribeBuffer(resources.workspace, layout.ssim),
            DescribeBuffer(resources.workspace, layout.mu1),
            DescribeBuffer(resources.workspace, layout.mu2),
            DescribeBuffer(resources.workspace, layout.var1),
            DescribeBuffer(resources.workspace, layout.var2),
            DescribeBuffer(resources.workspace, layout.cov12),
        }};
        PushStorageDescriptors(session, session.stage0Shader, descriptors);
        PushParams(session, session.stage0Shader, params);
    } else {
        BindShader(session, session.stage0ScoreShader);
        const std::array<VkDescriptorBufferInfo, 3> descriptors = {{
            DescribeBuffer(resources.workspace, layout.lab1),
            DescribeBuffer(resources.workspace, layout.lab2),
            DescribeBuffer(resources.workspace, layout.ssim),
        }};
        PushStorageDescriptors(session, session.stage0ScoreShader, descriptors);
        PushParams(session, session.stage0ScoreShader, params);
    }
    vkCmdDispatch(session.commandBuffer, wgX, wgY, 1);
    EndTimestamp(session);
    MemoryBarrier(
        session.commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT);

    std::array<VkBufferCopy, 6> readbacks{};
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
    vkCmdCopyBuffer(
        session.commandBuffer,
        resources.workspace.buffer,
        resources.readback.buffer,
        readbackCount,
        readbacks.data());
    MemoryBarrier(
        session.commandBuffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_HOST_BIT,
        VK_ACCESS_2_HOST_READ_BIT);
    SubmitCommands(session);
    outputs.dispatchAndSubmit_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - dispatchStartedAt);

    outputs.width = width;
    outputs.height = height;
    outputs.elemCount = elemCount;
    const auto readbackStartedAt = std::chrono::steady_clock::now();
    WaitForSubmission(session);
    InvalidateMappedBuffer(resources.readback);
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
    BufferRegion upload1;
    BufferRegion upload2;
    VkDeviceSize workspaceBytes = 0;
    VkDeviceSize uploadBytes = 0;
};

BatchStageLayout BuildBatchStageLayout(
    const GpuSession& session,
    VkDeviceSize rgba8Bytes,
    VkDeviceSize inputBytes,
    VkDeviceSize downsampleBytes,
    VkDeviceSize f32Bytes) {
    ArenaBuilder workspace{
        .alignment = std::max<VkDeviceSize>(
            4u,
            session.physicalDeviceProperties.limits.minStorageBufferOffsetAlignment),
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
    layout.workspaceBytes = workspace.cursor;

    ArenaBuilder upload{.alignment = 4u};
    layout.upload1 = upload.Add(rgba8Bytes);
    layout.upload2 = upload.Add(rgba8Bytes);
    layout.uploadBytes = upload.cursor;
    return layout;
}

struct VideoImageViews {
    VkImageView y = VK_NULL_HANDLE;
    VkImageView uv = VK_NULL_HANDLE;
    VkImageLayout originalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags originalAccess = 0;
    std::uint32_t originalQueueFamily = VK_QUEUE_FAMILY_IGNORED;

    VkDevice device = VK_NULL_HANDLE;
    ~VideoImageViews() {
        if (device != VK_NULL_HANDLE) {
            if (y != VK_NULL_HANDLE) {
                vkDestroyImageView(device, y, nullptr);
            }
            if (uv != VK_NULL_HANDLE) {
                vkDestroyImageView(device, uv, nullptr);
            }
        }
    }
    VideoImageViews(const VideoImageViews&) = delete;
    VideoImageViews& operator=(const VideoImageViews&) = delete;
    VideoImageViews() = default;
    VideoImageViews(VideoImageViews&& other) noexcept
        : y(std::exchange(other.y, VK_NULL_HANDLE)),
          uv(std::exchange(other.uv, VK_NULL_HANDLE)),
          originalLayout(other.originalLayout),
          originalAccess(other.originalAccess),
          originalQueueFamily(other.originalQueueFamily),
          device(std::exchange(other.device, VK_NULL_HANDLE)) {}
};

VkImageView CreateVideoPlaneView(
    const GpuSession& session,
    VkImage image,
    VkFormat format,
    VkImageAspectFlags aspect) {
    const VkImageViewCreateInfo viewInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    VkImageView view = VK_NULL_HANDLE;
    VkCheck(vkCreateImageView(session.device, &viewInfo, nullptr, &view), "vkCreateImageView");
    return view;
}

VideoImageViews CreateVideoImageViews(
    const GpuSession& session,
    const VulkanVideoFrame& frame) {
    if (frame.vkFrame == nullptr || frame.framesContext == nullptr) {
        throw std::runtime_error("invalid Vulkan Video frame metadata");
    }
    const bool tenBit = frame.softwareFormat == AV_PIX_FMT_P010LE ||
                        frame.softwareFormat == AV_PIX_FMT_P010BE;
    const VkFormat planeFormat = tenBit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
    const VkFormat chromaFormat = tenBit ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R8G8_UNORM;

    VideoImageViews views;
    views.device = session.device;
    views.originalLayout = frame.vkFrame->layout[0];
    views.originalAccess = frame.vkFrame->access[0];
    views.originalQueueFamily = frame.vkFrame->queue_family[0];

    if (frame.vkFrame->img[1] != VK_NULL_HANDLE) {
        views.y = CreateVideoPlaneView(session, frame.vkFrame->img[0], planeFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        views.uv = CreateVideoPlaneView(session, frame.vkFrame->img[1], chromaFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    } else {
        views.y = CreateVideoPlaneView(
            session, frame.vkFrame->img[0], planeFormat, VK_IMAGE_ASPECT_PLANE_0_BIT);
        views.uv = CreateVideoPlaneView(
            session, frame.vkFrame->img[0], chromaFormat, VK_IMAGE_ASPECT_PLANE_1_BIT);
    }
    return views;
}

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
    if (levelCount == 0 || heights.size() != levelCount) {
        throw std::runtime_error("invalid batch dimensions");
    }

    constexpr VkDeviceSize kReadbackAlignment = 256u;
    std::vector<VkDeviceSize> outputOffsets(levelCount);
    std::vector<std::size_t> elemCounts(levelCount);
    VkDeviceSize outputBytesTotal = 0;
    for (std::size_t level = 0; level < levelCount; ++level) {
        const std::size_t elemCount =
            static_cast<std::size_t>(widths[level]) * static_cast<std::size_t>(heights[level]);
        if (elemCount == 0 || elemCount > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("invalid or oversized pyramid level");
        }
        elemCounts[level] = elemCount;
        outputOffsets[level] = outputBytesTotal;
        outputBytesTotal += AlignUp(elemCount * sizeof(float), kReadbackAlignment);
    }

    const std::size_t baseElemCount = elemCounts.front();
    const VkDeviceSize rgba8Bytes = baseElemCount * 4u;
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
    const VkDeviceSize inputBytes = baseElemCount * sizeof(LinearRgba);
    const VkDeviceSize downsampleBytes =
        ((levelCount > 1u) ? elemCounts[1] : 1u) * sizeof(LinearRgba);
    const VkDeviceSize baseF32Bytes = baseElemCount * sizeof(float);
    ValidateStorageRange(session, rgba8Bytes, "packed RGBA8 input buffer");
    ValidateStorageRange(session, inputBytes, "linear RGBA/LAB buffer");
    ValidateStorageRange(session, downsampleBytes, "downsample buffer");
    ValidateStorageRange(session, baseF32Bytes, "SSIM output buffer");
    const BatchStageLayout layout = BuildBatchStageLayout(
        session, rgba8Bytes, inputBytes, downsampleBytes, baseF32Bytes);

    std::vector<ScaleOutputs> outputs(levelCount);
    if (!session.batchStage0Resources ||
        session.batchStage0Resources->workspaceCapacity < layout.workspaceBytes ||
        session.batchStage0Resources->uploadCapacity < layout.uploadBytes ||
        session.batchStage0Resources->readbackCapacity < outputBytesTotal) {
        const auto startedAt = std::chrono::steady_clock::now();
        auto resources = std::make_unique<BatchStage0Resources>();
        resources->workspaceCapacity = layout.workspaceBytes;
        resources->uploadCapacity = layout.uploadBytes;
        resources->readbackCapacity = outputBytesTotal;
        resources->workspace = CreateBuffer(
            session,
            layout.workspaceBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        resources->upload = CreateBuffer(
            session,
            layout.uploadBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        resources->readback = CreateBuffer(
            session,
            outputBytesTotal,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        session.batchStage0Resources = std::move(resources);
        outputs.front().createBuffers_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startedAt);
    }
    BatchStage0Resources& resources = *session.batchStage0Resources;

    const auto writeStartedAt = std::chrono::steady_clock::now();
    if (!videoInput) {
        auto* upload = static_cast<std::uint8_t*>(resources.upload.mapped);
        std::memcpy(upload + layout.upload1.offset, input1.data(), rgba8Bytes);
        std::memcpy(upload + layout.upload2.offset, input2.data(), rgba8Bytes);
        FlushMappedBuffer(resources.upload);
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
        videoViews1 = std::make_unique<VideoImageViews>(CreateVideoImageViews(session, *video1));
        videoViews2 = std::make_unique<VideoImageViews>(CreateVideoImageViews(session, *video2));
        reinterpret_cast<AVVulkanFramesContext*>(video1->framesContext->hwctx)
            ->lock_frame(video1->framesContext, video1->vkFrame);
        reinterpret_cast<AVVulkanFramesContext*>(video2->framesContext->hwctx)
            ->lock_frame(video2->framesContext, video2->vkFrame);
    }
    BeginCommands(session);
    if (!videoInput) {
        const std::array<VkBufferCopy, 2> inputCopies = {{
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
        vkCmdCopyBuffer(
            session.commandBuffer,
            resources.upload.buffer,
            resources.workspace.buffer,
            static_cast<std::uint32_t>(inputCopies.size()),
            inputCopies.data());
        MemoryBarrier(
            session.commandBuffer,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    } else {
        RecordVideoImageBarrier(
            session,
            *video1,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            session.queueFamilyIndex,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        RecordVideoImageBarrier(
            session,
            *video2,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
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
    const auto baseWorkgroups =
        ComputeWorkgroupCounts(session, widths.front(), heights.front());
    const std::uint32_t baseWgX = baseWorkgroups[0];
    const std::uint32_t baseWgY = baseWorkgroups[1];
    const VkDescriptorBufferInfo lutDescriptor = {
        .buffer = session.srgbToLinearLutBuffer.buffer,
        .offset = 0,
        .range = 256u * sizeof(float),
    };
    if (!videoInput) {
        BindShader(session, session.rgba8ToLinearShader);
        const std::array<VkDescriptorBufferInfo, 3> convert1 = {{
            DescribeBuffer(resources.workspace, layout.rgba8Input1),
            DescribeBuffer(resources.workspace, layout.input1),
            lutDescriptor,
        }};
        PushStorageDescriptors(session, session.rgba8ToLinearShader, convert1);
        PushParams(session, session.rgba8ToLinearShader, baseParams);
        vkCmdDispatch(session.commandBuffer, baseWgX, baseWgY, 1);
        const std::array<VkDescriptorBufferInfo, 3> convert2 = {{
            DescribeBuffer(resources.workspace, layout.rgba8Input2),
            DescribeBuffer(resources.workspace, layout.input2),
            lutDescriptor,
        }};
        PushStorageDescriptors(session, session.rgba8ToLinearShader, convert2);
        PushParams(session, session.rgba8ToLinearShader, baseParams);
        vkCmdDispatch(session.commandBuffer, baseWgX, baseWgY, 1);
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
        BindShader(session, session.vulkanYuvToRgbaShader);
        const std::array<VkDescriptorImageInfo, 2> images1 = {{
            {.sampler = session.videoYSampler, .imageView = videoViews1->y, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {.sampler = session.videoUvSampler, .imageView = videoViews1->uv, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        }};
        PushSampledImageDescriptors(session, session.vulkanYuvToRgbaShader, images1);
        const VkDescriptorBufferInfo output1 = DescribeBuffer(resources.workspace, layout.rgba8Input1);
        PushStorageDescriptors(session, session.vulkanYuvToRgbaShader, std::span(&output1, 1), 2);
        PushParams(session, session.vulkanYuvToRgbaShader, yuvParams);
        vkCmdDispatch(session.commandBuffer, baseWgX, baseWgY, 1);
        const std::array<VkDescriptorImageInfo, 2> images2 = {{
            {.sampler = session.videoYSampler, .imageView = videoViews2->y, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {.sampler = session.videoUvSampler, .imageView = videoViews2->uv, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        }};
        PushSampledImageDescriptors(session, session.vulkanYuvToRgbaShader, images2);
        const VkDescriptorBufferInfo output2 = DescribeBuffer(resources.workspace, layout.rgba8Input2);
        PushStorageDescriptors(session, session.vulkanYuvToRgbaShader, std::span(&output2, 1), 2);
        PushParams(session, session.vulkanYuvToRgbaShader, yuvParams);
        vkCmdDispatch(session.commandBuffer, baseWgX, baseWgY, 1);
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

        // Keep the same packed-RGBA8 -> linear-premultiplied path as still
        // images. The YUV conversion above only replaces the CPU upload.
        BindShader(session, session.rgba8ToLinearShader);
        const std::array<VkDescriptorBufferInfo, 3> convert1 = {{
            DescribeBuffer(resources.workspace, layout.rgba8Input1),
            DescribeBuffer(resources.workspace, layout.input1),
            lutDescriptor,
        }};
        PushStorageDescriptors(session, session.rgba8ToLinearShader, convert1);
        PushParams(session, session.rgba8ToLinearShader, baseParams);
        vkCmdDispatch(session.commandBuffer, baseWgX, baseWgY, 1);
        const std::array<VkDescriptorBufferInfo, 3> convert2 = {{
            DescribeBuffer(resources.workspace, layout.rgba8Input2),
            DescribeBuffer(resources.workspace, layout.input2),
            lutDescriptor,
        }};
        PushStorageDescriptors(session, session.rgba8ToLinearShader, convert2);
        PushParams(session, session.rgba8ToLinearShader, baseParams);
        vkCmdDispatch(session.commandBuffer, baseWgX, baseWgY, 1);
    }
    MemoryBarrier(
        session.commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    for (std::size_t level = 0; level < levelCount; ++level) {
        const VkDeviceSize rgbaBytes = elemCounts[level] * sizeof(LinearRgba);
        const VkDeviceSize f32Bytes = elemCounts[level] * sizeof(float);
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

        BindShader(session, session.preprocessShader);
        const std::array<VkDescriptorBufferInfo, 2> preprocess1 = {{
            DescribeBuffer(resources.workspace, currentInput1),
            DescribeBuffer(resources.workspace, currentLab1),
        }};
        PushStorageDescriptors(session, session.preprocessShader, preprocess1);
        PushParams(session, session.preprocessShader, params);
        vkCmdDispatch(session.commandBuffer, wgX, wgY, 1);
        const std::array<VkDescriptorBufferInfo, 2> preprocess2 = {{
            DescribeBuffer(resources.workspace, currentInput2),
            DescribeBuffer(resources.workspace, currentLab2),
        }};
        PushStorageDescriptors(session, session.preprocessShader, preprocess2);
        PushParams(session, session.preprocessShader, params);
        vkCmdDispatch(session.commandBuffer, wgX, wgY, 1);
        MemoryBarrier(
            session.commandBuffer,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

        BindShader(session, session.stage0ScoreShader);
        const std::array<VkDescriptorBufferInfo, 3> scoreDescriptors = {{
            DescribeBuffer(resources.workspace, currentLab1),
            DescribeBuffer(resources.workspace, currentLab2),
            DescribeBuffer(resources.workspace, currentSsim),
        }};
        PushStorageDescriptors(session, session.stage0ScoreShader, scoreDescriptors);
        PushParams(session, session.stage0ScoreShader, params);
        vkCmdDispatch(session.commandBuffer, wgX, wgY, 1);
        if (level + 1u == levelCount) {
            EndTimestamp(session);
        }
        MemoryBarrier(
            session.commandBuffer,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT);
        const VkBufferCopy scoreCopy = {
            .srcOffset = currentSsim.offset,
            .dstOffset = outputOffsets[level],
            .size = f32Bytes,
        };
        vkCmdCopyBuffer(
            session.commandBuffer,
            resources.workspace.buffer,
            resources.readback.buffer,
            1,
            &scoreCopy);

        if (level + 1u < levelCount) {
            const VkDeviceSize nextRgbaBytes = elemCounts[level + 1u] * sizeof(LinearRgba);
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
            BindShader(session, session.downsampleShader);
            const std::array<VkDescriptorBufferInfo, 2> downsample1 = {{
                DescribeBuffer(resources.workspace, currentInput1),
                DescribeBuffer(resources.workspace, nextInput1),
            }};
            PushStorageDescriptors(session, session.downsampleShader, downsample1);
            PushParams(session, session.downsampleShader, downsampleParams);
            vkCmdDispatch(session.commandBuffer, downsampleWgX, downsampleWgY, 1);
            const std::array<VkDescriptorBufferInfo, 2> downsample2 = {{
                DescribeBuffer(resources.workspace, currentInput2),
                DescribeBuffer(resources.workspace, nextInput2),
            }};
            PushStorageDescriptors(session, session.downsampleShader, downsample2);
            PushParams(session, session.downsampleShader, downsampleParams);
            vkCmdDispatch(session.commandBuffer, downsampleWgX, downsampleWgY, 1);
            MemoryBarrier(
                session.commandBuffer,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
    }

    MemoryBarrier(
        session.commandBuffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_HOST_BIT,
        VK_ACCESS_2_HOST_READ_BIT);
    const std::array<const VulkanVideoFrame*, 2> submittedVideoFrames = {video1, video2};
    SubmitCommands(
        session,
        videoInput ? std::span<const VulkanVideoFrame* const>(submittedVideoFrames) :
                      std::span<const VulkanVideoFrame* const>());
    if (videoInput) {
        reinterpret_cast<AVVulkanFramesContext*>(video1->framesContext->hwctx)
            ->unlock_frame(video1->framesContext, video1->vkFrame);
        reinterpret_cast<AVVulkanFramesContext*>(video2->framesContext->hwctx)
            ->unlock_frame(video2->framesContext, video2->vkFrame);
    }
    outputs.front().dispatchAndSubmit_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - dispatchStartedAt);

    const auto readbackStartedAt = std::chrono::steady_clock::now();
    WaitForSubmission(session);
    InvalidateMappedBuffer(resources.readback);
    const auto* ssimBytes = static_cast<const std::uint8_t*>(resources.readback.mapped);
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
        const float* ssimValues = reinterpret_cast<const float*>(
            ssimBytes + static_cast<std::size_t>(outputOffsets[level]));
        const double ssimSum = SumF32(ssimValues, elemCounts[level]);
        output.meanSsim = ssimSum / static_cast<double>(elemCounts[level]);
        const double avg = std::pow(
            std::max(output.meanSsim, 0.0),
            std::pow(0.5, static_cast<double>(level)));
        const double devSum =
            SumAbsoluteDeviation(ssimValues, elemCounts[level], avg);
        output.ssimScore = 1.0 - devSum / static_cast<double>(elemCounts[level]);
    };
    if (levelCount > 1u && baseElemCount >= 65536u) {
        auto remainingLevels = std::async(std::launch::async, [&] {
            const auto startedAt = std::chrono::steady_clock::now();
            for (std::size_t level = 1; level < levelCount; ++level) {
                processLevel(level);
            }
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - startedAt)
                .count();
        });
        const auto baseScaleStartedAt = std::chrono::steady_clock::now();
        processLevel(0);
        outputs.front().postProcessBaseScaleMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - baseScaleStartedAt)
                .count();
        outputs.front().postProcessRemainingScalesMs = remainingLevels.get();
    } else {
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

struct PhysicalDeviceSelection {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    std::uint32_t queueFamilyIndex = 0;
    VkQueueFamilyProperties queueFamilyProperties{};
    std::uint32_t videoDecodeQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    VkQueueFamilyProperties videoDecodeQueueProperties{};
    VkVideoCodecOperationFlagsKHR videoDecodeCaps = 0;
    std::uint32_t videoDecodeQueueFamilyIndexSecondary = VK_QUEUE_FAMILY_IGNORED;
    VkQueueFamilyProperties videoDecodeQueuePropertiesSecondary{};
    VkVideoCodecOperationFlagsKHR videoDecodeCapsSecondary = 0;
    VkPhysicalDeviceProperties properties{};
};

PhysicalDeviceSelection SelectPhysicalDevice(VkInstance instance) {
    std::uint32_t deviceCount = 0;
    VkCheck(
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr),
        "vkEnumeratePhysicalDevices(count)");
    if (deviceCount == 0) {
        throw std::runtime_error("no Vulkan physical device found");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    VkCheck(
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()),
        "vkEnumeratePhysicalDevices");

    PhysicalDeviceSelection best;
    int bestScore = std::numeric_limits<int>::min();
    for (VkPhysicalDevice physicalDevice : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        if (VK_API_VERSION_MAJOR(properties.apiVersion) < 1u ||
            (VK_API_VERSION_MAJOR(properties.apiVersion) == 1u &&
             VK_API_VERSION_MINOR(properties.apiVersion) < 3u)) {
            continue;
        }
        const VkPhysicalDeviceLimits& limits = properties.limits;
        if (limits.maxComputeWorkGroupInvocations < 16u * 16u ||
            limits.maxComputeWorkGroupSize[0] < 16u ||
            limits.maxComputeWorkGroupSize[1] < 16u ||
            limits.maxComputeSharedMemorySize < 20u * 20u * 4u * sizeof(float) ||
            limits.maxPerStageDescriptorStorageBuffers < 8u) {
            continue;
        }
        if (!HasDeviceExtension(physicalDevice, VK_EXT_SHADER_OBJECT_EXTENSION_NAME) ||
            !HasDeviceExtension(physicalDevice, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
            continue;
        }
        const bool hasVideoDecodeExtensions =
            HasDeviceExtension(physicalDevice, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) &&
            HasDeviceExtension(physicalDevice, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME) &&
            (HasDeviceExtension(physicalDevice, VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME) ||
             HasDeviceExtension(physicalDevice, VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME) ||
             HasDeviceExtension(physicalDevice, VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME) ||
             HasDeviceExtension(physicalDevice, VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME));

        VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{};
        shaderObjectFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
        VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13Features.pNext = &shaderObjectFeatures;
        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &vulkan13Features;
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
        if (shaderObjectFeatures.shaderObject != VK_TRUE ||
            vulkan13Features.synchronization2 != VK_TRUE ||
            vulkan13Features.dynamicRendering != VK_TRUE) {
            continue;
        }

        std::uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            physicalDevice, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(
            physicalDevice, &queueCount, queues.data());
        std::uint32_t videoQueueIndex = VK_QUEUE_FAMILY_IGNORED;
        VkQueueFamilyProperties videoQueueProperties{};
        for (std::uint32_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
            const VkQueueFamilyProperties& queue = queues[queueIndex];
            if (queue.queueCount == 0) {
                continue;
            }
            if (hasVideoDecodeExtensions &&
                (queue.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) != 0 &&
                videoQueueIndex == VK_QUEUE_FAMILY_IGNORED) {
                videoQueueIndex = queueIndex;
                videoQueueProperties = queue;
            }
            if ((queue.queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
                continue;
            }
            int score = 0;
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                score += 1000;
            } else if (properties.deviceType ==
                       VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                score += 500;
            }
            if ((queue.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
                score += 100;
            }
            if (score > bestScore) {
                bestScore = score;
                best = {
                    .physicalDevice = physicalDevice,
                    .queueFamilyIndex = queueIndex,
                    .queueFamilyProperties = queue,
                    .videoDecodeQueueFamilyIndex = videoQueueIndex,
                    .videoDecodeQueueProperties = videoQueueProperties,
                    .properties = properties,
                };
            }
        }
    }
    if (best.physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "no Vulkan 1.3 compute device supports VK_EXT_shader_object, "
            "VK_KHR_push_descriptor, synchronization2, dynamic rendering, "
            "and the required compute shader limits");
    }
    if (HasDeviceExtension(best.physicalDevice, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) &&
        HasDeviceExtension(best.physicalDevice, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME) &&
        (HasDeviceExtension(best.physicalDevice, VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME) ||
         HasDeviceExtension(best.physicalDevice, VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME) ||
         HasDeviceExtension(best.physicalDevice, VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME) ||
         HasDeviceExtension(best.physicalDevice, VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME))) {
        std::uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties2(best.physicalDevice, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties2> queues(queueCount);
        std::vector<VkQueueFamilyVideoPropertiesKHR> videoProperties(queueCount);
        for (std::uint32_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
            queues[queueIndex].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
            videoProperties[queueIndex].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
            queues[queueIndex].pNext = &videoProperties[queueIndex];
        }
        vkGetPhysicalDeviceQueueFamilyProperties2(
            best.physicalDevice,
            &queueCount,
            queues.data());
        best.videoDecodeQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        best.videoDecodeQueueFamilyIndexSecondary = VK_QUEUE_FAMILY_IGNORED;
        for (std::uint32_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
            if (queues[queueIndex].queueFamilyProperties.queueCount == 0 ||
                (queues[queueIndex].queueFamilyProperties.queueFlags &
                 VK_QUEUE_VIDEO_DECODE_BIT_KHR) == 0 ||
                videoProperties[queueIndex].videoCodecOperations == 0) {
                continue;
            }
            if (best.videoDecodeQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED) {
                best.videoDecodeQueueFamilyIndex = queueIndex;
                best.videoDecodeQueueProperties = queues[queueIndex].queueFamilyProperties;
                best.videoDecodeCaps = videoProperties[queueIndex].videoCodecOperations;
            } else if (best.videoDecodeQueueFamilyIndexSecondary == VK_QUEUE_FAMILY_IGNORED) {
                best.videoDecodeQueueFamilyIndexSecondary = queueIndex;
                best.videoDecodeQueuePropertiesSecondary =
                    queues[queueIndex].queueFamilyProperties;
                best.videoDecodeCapsSecondary = videoProperties[queueIndex].videoCodecOperations;
            }
        }
    }
    if (best.properties.limits.maxPushConstantsSize < 16u) {
        throw std::runtime_error(
            "selected Vulkan device has less than 16 bytes of push constants");
    }

    VkPhysicalDevicePushDescriptorPropertiesKHR pushProperties{};
    pushProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &pushProperties;
    vkGetPhysicalDeviceProperties2(best.physicalDevice, &properties2);
    if (pushProperties.maxPushDescriptors < 8u) {
        throw std::runtime_error(
            "selected Vulkan device supports fewer than 8 push descriptors");
    }
    return best;
}

ComputeShader CreateComputeShader(
    GpuSession& session,
    std::span<const std::uint32_t> spirv,
    std::uint32_t descriptorCount,
    std::span<const VkDescriptorType> descriptorTypes = {}) {
    ComputeShader result;
    try {
        if (!descriptorTypes.empty() && descriptorTypes.size() != descriptorCount) {
            throw std::runtime_error("descriptor type count does not match shader bindings");
        }
        std::vector<VkDescriptorSetLayoutBinding> bindings(descriptorCount);
        for (std::uint32_t binding = 0; binding < descriptorCount; ++binding) {
            const VkDescriptorType descriptorType = descriptorTypes.empty()
                                                        ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                                        : descriptorTypes[binding];
            bindings[binding] = {
                .binding = binding,
                .descriptorType = descriptorType,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            };
        }
        const VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
            .bindingCount = descriptorCount,
            .pBindings = bindings.data(),
        };
        VkCheck(
            vkCreateDescriptorSetLayout(
                session.device,
                &descriptorLayoutInfo,
                nullptr,
                &result.descriptorSetLayout),
            "vkCreateDescriptorSetLayout");

        const VkPushConstantRange pushConstantRange = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = 4u * sizeof(std::uint32_t),
        };
        const VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &result.descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
        };
        VkCheck(
            vkCreatePipelineLayout(
                session.device,
                &pipelineLayoutInfo,
                nullptr,
                &result.pipelineLayout),
            "vkCreatePipelineLayout");

        const VkShaderCreateInfoEXT shaderInfo = {
            .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .nextStage = 0,
            .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
            .codeSize = spirv.size_bytes(),
            .pCode = spirv.data(),
            .pName = "main",
            .setLayoutCount = 1,
            .pSetLayouts = &result.descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
        };
        VkCheck(
            session.createShaders(
                session.device, 1, &shaderInfo, nullptr, &result.shader),
            "vkCreateShadersEXT");
    } catch (...) {
        DestroyComputeShader(session, result);
        throw;
    }
    return result;
}

std::unique_ptr<GpuSession> CreateGpuSession(
    std::span<const std::uint32_t> labPreprocessSpirv,
    std::span<const std::uint32_t> stage0Spirv,
    std::span<const std::uint32_t> stage0ScoreSpirv,
    std::span<const std::uint32_t> downsampleSpirv,
    std::span<const std::uint32_t> rgba8ToLinearSpirv,
    std::span<const std::uint32_t> vulkanYuvToRgbaSpirv,
    bool enableDebugPipeline,
    bool enableTimestampQueries) {
    auto session = std::make_unique<GpuSession>();

    std::uint32_t loaderVersion = VK_API_VERSION_1_0;
    VkCheck(vkEnumerateInstanceVersion(&loaderVersion), "vkEnumerateInstanceVersion");
    if (loaderVersion < VK_API_VERSION_1_3) {
        throw std::runtime_error("Vulkan loader 1.3 or newer is required");
    }
    const VkApplicationInfo applicationInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "dssim-WebGPU",
        .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .pEngineName = "dssim-vulkan",
        .engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo instanceInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &applicationInfo,
    };
    VkCheck(
        vkCreateInstance(&instanceInfo, nullptr, &session->instance),
        "vkCreateInstance");

    const PhysicalDeviceSelection selection =
        SelectPhysicalDevice(session->instance);
    session->physicalDevice = selection.physicalDevice;
    session->queueFamilyIndex = selection.queueFamilyIndex;
    session->computeQueueFlags = selection.queueFamilyProperties.queueFlags;
    session->physicalDeviceProperties = selection.properties;
    session->adapterName = selection.properties.deviceName;
    session->videoDecodeQueueFamilyIndex = selection.videoDecodeQueueFamilyIndex;
    session->videoDecodeQueueFlags = selection.videoDecodeQueueProperties.queueFlags;
    const auto maskEnabledVideoDecodeExtensions = [&](VkVideoCodecOperationFlagsKHR caps) {
        if (!HasDeviceExtension(
                session->physicalDevice,
                VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME)) {
            caps &= ~VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
        }
        if (!HasDeviceExtension(
                session->physicalDevice,
                VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME)) {
            caps &= ~VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
        }
        if (!HasDeviceExtension(
                session->physicalDevice,
                VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME)) {
            caps &= ~VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
        }
        if (!HasDeviceExtension(
                session->physicalDevice,
                VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME)) {
            caps &= ~VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
        }
        return caps;
    };
    session->videoDecodeCaps = maskEnabledVideoDecodeExtensions(selection.videoDecodeCaps);
    session->videoDecodeQueueFamilyIndexSecondary =
        selection.videoDecodeQueueFamilyIndexSecondary;
    session->videoDecodeQueueFlagsSecondary =
        selection.videoDecodeQueuePropertiesSecondary.queueFlags;
    session->videoDecodeCapsSecondary =
        maskEnabledVideoDecodeExtensions(selection.videoDecodeCapsSecondary);
    session->videoSupported =
        selection.videoDecodeQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED &&
        session->videoDecodeCaps != 0;

    VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{};
    shaderObjectFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
    shaderObjectFeatures.shaderObject = VK_TRUE;
    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &shaderObjectFeatures;
    vulkan13Features.synchronization2 = VK_TRUE;
    vulkan13Features.dynamicRendering = VK_TRUE;

    const float queuePriority = 1.0f;
    const VkDeviceQueueCreateInfo computeQueueInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = session->queueFamilyIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };
    std::array<VkDeviceQueueCreateInfo, 3> queueInfos = {computeQueueInfo, {}, {}};
    std::uint32_t queueInfoCount = 1;
    if (session->videoSupported &&
        session->videoDecodeQueueFamilyIndex != session->queueFamilyIndex) {
        queueInfos[queueInfoCount++] = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = session->videoDecodeQueueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        };
    }
    if (session->videoSupported &&
        session->videoDecodeQueueFamilyIndexSecondary != VK_QUEUE_FAMILY_IGNORED &&
        session->videoDecodeQueueFamilyIndexSecondary != session->queueFamilyIndex &&
        session->videoDecodeQueueFamilyIndexSecondary != session->videoDecodeQueueFamilyIndex) {
        queueInfos[queueInfoCount++] = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = session->videoDecodeQueueFamilyIndexSecondary,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        };
    }
    std::vector<const char*> deviceExtensions = {
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
    };
    if (session->videoSupported) {
        deviceExtensions.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);
        const VkVideoCodecOperationFlagsKHR allVideoDecodeCaps =
            session->videoDecodeCaps | session->videoDecodeCapsSecondary;
        if ((allVideoDecodeCaps & VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR) != 0) {
            deviceExtensions.push_back(VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME);
        }
        if ((allVideoDecodeCaps & VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR) != 0) {
            deviceExtensions.push_back(VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME);
        }
        if ((allVideoDecodeCaps & VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR) != 0) {
            deviceExtensions.push_back(VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME);
        }
        if ((allVideoDecodeCaps & VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR) != 0) {
            deviceExtensions.push_back(VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME);
        }
        if (HasDeviceExtension(
                session->physicalDevice,
                VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME)) {
            deviceExtensions.push_back(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);
        }
        session->videoDeviceExtensions = deviceExtensions;
    }
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &vulkan13Features;
    deviceInfo.queueCreateInfoCount = queueInfoCount;
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.enabledExtensionCount =
        static_cast<std::uint32_t>(deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
    VkCheck(
        vkCreateDevice(
            session->physicalDevice, &deviceInfo, nullptr, &session->device),
        "vkCreateDevice");
    vkGetDeviceQueue(
        session->device,
        session->queueFamilyIndex,
        0,
        &session->queue);

    session->createShaders = reinterpret_cast<PFN_vkCreateShadersEXT>(
        vkGetDeviceProcAddr(session->device, "vkCreateShadersEXT"));
    session->destroyShader = reinterpret_cast<PFN_vkDestroyShaderEXT>(
        vkGetDeviceProcAddr(session->device, "vkDestroyShaderEXT"));
    session->cmdBindShaders = reinterpret_cast<PFN_vkCmdBindShadersEXT>(
        vkGetDeviceProcAddr(session->device, "vkCmdBindShadersEXT"));
    session->cmdPushDescriptorSet =
        reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
            vkGetDeviceProcAddr(session->device, "vkCmdPushDescriptorSetKHR"));
    if (session->createShaders == nullptr || session->destroyShader == nullptr ||
        session->cmdBindShaders == nullptr ||
        session->cmdPushDescriptorSet == nullptr) {
        throw std::runtime_error(
            "Vulkan driver did not expose required shader object/push descriptor commands");
    }

    const auto resourceStartedAt = std::chrono::steady_clock::now();
    const VkCommandPoolCreateInfo commandPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = session->queueFamilyIndex,
    };
    VkCheck(
        vkCreateCommandPool(
            session->device,
            &commandPoolInfo,
            nullptr,
            &session->commandPool),
        "vkCreateCommandPool");
    const VkCommandBufferAllocateInfo commandBufferInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = session->commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCheck(
        vkAllocateCommandBuffers(
            session->device,
            &commandBufferInfo,
            &session->commandBuffer),
        "vkAllocateCommandBuffers");
    const VkFenceCreateInfo fenceInfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    VkCheck(
        vkCreateFence(
            session->device, &fenceInfo, nullptr, &session->submitFence),
        "vkCreateFence");

    session->timestampQueryEnabled =
        enableTimestampQueries &&
        selection.queueFamilyProperties.timestampValidBits != 0u;
    session->timestampValidBits =
        session->timestampQueryEnabled
            ? selection.queueFamilyProperties.timestampValidBits
            : 0u;
    if (session->timestampQueryEnabled) {
        const VkQueryPoolCreateInfo queryPoolInfo = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
        };
        VkCheck(
            vkCreateQueryPool(
                session->device,
                &queryPoolInfo,
                nullptr,
                &session->timestampQueryPool),
            "vkCreateQueryPool");
    }

    session->srgbToLinearLutBuffer = CreateBuffer(
        *session,
        256u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const auto& lut = SrgbToLinearLut();
    std::memcpy(
        session->srgbToLinearLutBuffer.mapped,
        lut.data(),
        lut.size() * sizeof(float));
    FlushMappedBuffer(session->srgbToLinearLutBuffer);
    session->initProfiling.createBuffersTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - resourceStartedAt);

    const auto shadersStartedAt = std::chrono::steady_clock::now();
    session->rgba8ToLinearShader =
        CreateComputeShader(*session, rgba8ToLinearSpirv, 3);
    session->preprocessShader =
        CreateComputeShader(*session, labPreprocessSpirv, 2);
    session->stage0ScoreShader =
        CreateComputeShader(*session, stage0ScoreSpirv, 3);
    if (enableDebugPipeline) {
        session->stage0Shader =
            CreateComputeShader(*session, stage0Spirv, 8);
    }
    session->downsampleShader =
        CreateComputeShader(*session, downsampleSpirv, 2);
    if (session->videoSupported) {
        const std::array<VkDescriptorType, 3> videoDescriptorTypes = {
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        session->vulkanYuvToRgbaShader = CreateComputeShader(
            *session,
            vulkanYuvToRgbaSpirv,
            3,
            videoDescriptorTypes);
        const VkSamplerCreateInfo ySamplerInfo = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .minLod = 0.0f,
            .maxLod = 0.0f,
        };
        const VkSamplerCreateInfo uvSamplerInfo = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .minLod = 0.0f,
            .maxLod = 0.0f,
        };
        VkCheck(vkCreateSampler(session->device, &ySamplerInfo, nullptr, &session->videoYSampler), "vkCreateSampler(y)");
        VkCheck(vkCreateSampler(session->device, &uvSamplerInfo, nullptr, &session->videoUvSampler), "vkCreateSampler(uv)");
    }
    session->initProfiling.createShaderModuleTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - shadersStartedAt);
    return session;
}

ComparisonRequest ParseComparisonRequestLine(const std::string& line) {
    const std::size_t separator = line.find('\t');
    if (separator == std::string::npos) {
        throw std::runtime_error("stdin pair line must be tab-delimited: <img1>\\t<img2>");
    }
    if (separator == 0 || separator + 1 >= line.size()) {
        throw std::runtime_error("stdin pair line contains an empty image path");
    }
    return {
        .image1 = line.substr(0, separator),
        .image2 = line.substr(separator + 1),
    };
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

    // Identical input produces score 0 by definition (matches reference behavior).
    // GPU dispatches may introduce f32 non-determinism between separate runs,
    // so we detect this case on the CPU side.
    const bool identicalInput = !videoInput && std::equal(rgba1.begin(), rgba1.end(), rgba2.begin());

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

    if (identicalInput) {
        compute.weightedSsim = 1.0;
        compute.score = 0.0;
    } else {
        double weightedSum = 0.0;
        double weightTotal = 0.0;
        for (std::size_t i = 0; i < compute.scales.size(); ++i) {
            const double w = kDefaultScaleWeights[i];
            weightedSum += compute.scales[i].ssimScore * w;
            weightTotal += w;
        }
        compute.weightedSsim = weightedSum / weightTotal;
        compute.score = 1.0 / std::max(compute.weightedSsim, std::numeric_limits<double>::epsilon()) - 1.0;
    }

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
    const DecodedImage image1 = LoadPngRgba8(request.image1);
    const DecodedImage image2 = LoadPngRgba8(request.image2);
    if (image1.pixels.empty() || image2.pixels.empty()) {
        throw std::runtime_error("decoded png pixels are empty");
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
        const auto labPreprocessShaderPath = ResolveShaderPath(argv[0], "lab_preprocess.spv");
        const auto downsampleShaderPath = ResolveShaderPath(argv[0], "downsample_2x2.spv");
        const auto rgba8ToLinearShaderPath = ResolveShaderPath(argv[0], "rgba8_to_linear.spv");
        const auto vulkanYuvToRgbaShaderPath = ResolveShaderPath(argv[0], "vulkan_yuv_to_rgba8.spv");
        const auto stage0Spirv = ReadSpirv(stage0ShaderPath);
        const auto stage0ScoreSpirv = ReadSpirv(stage0ScoreShaderPath);
        const auto labPreprocessSpirv = ReadSpirv(labPreprocessShaderPath);
        const auto downsampleSpirv = ReadSpirv(downsampleShaderPath);
        const auto rgba8ToLinearSpirv = ReadSpirv(rgba8ToLinearShaderPath);
        const auto vulkanYuvToRgbaSpirv = ReadSpirv(vulkanYuvToRgbaShaderPath);
        auto session =
            CreateGpuSession(
                labPreprocessSpirv,
                stage0Spirv,
                stage0ScoreSpirv,
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
        std::cerr << "dssim-WebGPU error: " << ex.what() << '\n';
        return 1;
    }
}

