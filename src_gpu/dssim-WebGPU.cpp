#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <numeric>
#include <span>

#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp.h>

#include "png_loader.h"
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
    std::filesystem::path debugDumpDir;
    bool debugDumpEnabled = false;
    bool stdinPairsMode = false;
    bool profilingEnabled = false;
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
};

struct RgbaPairComparisonResult {
    MultiScaleOutputs compute;
    ProfilingSummary profiling;
    std::vector<LinearRgba> debugScale1Image1;
    std::vector<LinearRgba> debugScale1Image2;
};

struct Stage0Resources {
    std::size_t capacity = 0;
    wgpu::Buffer input1Buffer;
    wgpu::Buffer input2Buffer;
    wgpu::Buffer lab1Buffer;
    wgpu::Buffer lab2Buffer;
    wgpu::Buffer outSsimBuffer;
    wgpu::Buffer outMu1Buffer;
    wgpu::Buffer outMu2Buffer;
    wgpu::Buffer outVar1Buffer;
    wgpu::Buffer outVar2Buffer;
    wgpu::Buffer outCov12Buffer;
    wgpu::Buffer readbackSsimBuffer;
    wgpu::Buffer readbackMu1Buffer;
    wgpu::Buffer readbackMu2Buffer;
    wgpu::Buffer readbackVar1Buffer;
    wgpu::Buffer readbackVar2Buffer;
    wgpu::Buffer readbackCov12Buffer;
    wgpu::Buffer paramsBuffer;
    wgpu::BindGroup preprocessBindGroup1;
    wgpu::BindGroup preprocessBindGroup2;
    wgpu::BindGroup stage0BindGroup;
};

struct BatchStage0Resources {
    std::size_t rgba8CapacityBytes = 0;
    std::size_t inputCapacityBytes = 0;
    std::size_t downsampleCapacityBytes = 0;
    std::size_t outputCapacityBytes = 0;
    std::size_t baseCapacity = 0;
    wgpu::Buffer rgba8Input1Buffer;
    wgpu::Buffer rgba8Input2Buffer;
    wgpu::Buffer input1Buffer;
    wgpu::Buffer input2Buffer;
    wgpu::Buffer downsample1Buffer;
    wgpu::Buffer downsample2Buffer;
    wgpu::Buffer lab1Buffer;
    wgpu::Buffer lab2Buffer;
    wgpu::Buffer outSsimBuffer;
    wgpu::Buffer outMu1Buffer;
    wgpu::Buffer outMu2Buffer;
    wgpu::Buffer outVar1Buffer;
    wgpu::Buffer outVar2Buffer;
    wgpu::Buffer outCov12Buffer;
    wgpu::Buffer readbackSsimBuffer;
    wgpu::Buffer paramsBuffer;
    wgpu::Buffer downsampleParamsBuffer;
    wgpu::Buffer srgbToLinearLutBuffer;
};

struct GpuSession {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    std::string adapterName = "unknown";
    bool timestampQueryEnabled = false;
    wgpu::QuerySet timestampQuerySet;
    wgpu::Buffer timestampResolveBuffer;
    wgpu::Buffer timestampReadbackBuffer;

    wgpu::ShaderModule preprocessShader;
    wgpu::ShaderModule stage0Shader;
    wgpu::ShaderModule stage0ScoreShader;
    wgpu::ShaderModule downsampleShader;
    wgpu::ShaderModule rgba8ToLinearShader;

    wgpu::BindGroupLayout rgba8ToLinearBindGroupLayout;
    wgpu::PipelineLayout rgba8ToLinearPipelineLayout;
    wgpu::ComputePipeline rgba8ToLinearPipeline;

    wgpu::BindGroupLayout preprocessBindGroupLayout;
    wgpu::PipelineLayout preprocessPipelineLayout;
    wgpu::ComputePipeline preprocessPipeline;

    wgpu::BindGroupLayout stage0BindGroupLayout;
    wgpu::PipelineLayout stage0PipelineLayout;
    wgpu::ComputePipeline stage0Pipeline;

    wgpu::BindGroupLayout stage0ScoreBindGroupLayout;
    wgpu::PipelineLayout stage0ScorePipelineLayout;
    wgpu::ComputePipeline stage0ScorePipeline;

    wgpu::BindGroupLayout downsampleBindGroupLayout;
    wgpu::PipelineLayout downsamplePipelineLayout;
    wgpu::ComputePipeline downsamplePipeline;

    std::unique_ptr<Stage0Resources> stage0Resources;
    std::unique_ptr<Stage0Resources> debugStage0Resources;
    std::unique_ptr<BatchStage0Resources> batchStage0Resources;

    ProfilingSummary initProfiling;
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

std::string ReadAllText(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open text file: " + path.string());
    }

    std::ostringstream oss;
    oss << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("failed to read text file: " + path.string());
    }
    return oss.str();
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
    if (options.stdinPairsMode) {
        if (positionalCount != 0) {
            throw std::runtime_error("--stdin-pairs does not accept positional image arguments");
        }
        if (!options.out.empty()) {
            throw std::runtime_error("--stdin-pairs cannot be combined with --out");
        }
        if (options.debugDumpEnabled) {
            throw std::runtime_error("--stdin-pairs cannot be combined with --debug-dump-dir");
        }
    } else if (positionalCount != 2) {
        throw std::runtime_error(
            "usage: dssim-WebGPU <img1> <img2> [--out <json>] "
            "[--debug-dump-dir <dir>] [--stdin-pairs] [--profiling]");
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

void WriteU32LeBuffer(const std::filesystem::path& outPath, const std::vector<std::uint32_t>& values) {
    const auto parent = outPath.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open output: " + outPath.string());
    }

    for (std::uint32_t v : values) {
        const std::uint8_t bytes[4] = {
            static_cast<std::uint8_t>(v & 0xFFu),
            static_cast<std::uint8_t>((v >> 8) & 0xFFu),
            static_cast<std::uint8_t>((v >> 16) & 0xFFu),
            static_cast<std::uint8_t>((v >> 24) & 0xFFu),
        };
        out.write(reinterpret_cast<const char*>(bytes), 4);
    }

    if (!out) {
        throw std::runtime_error("failed to write output: " + outPath.string());
    }
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
    os << "  \"engine\": \"gpu-dawn-wgsl-dssim-ms-stage5x5-gaussian-linear\",\n";
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
    os << "  \"version\": \"dawn-dssim-ms-stage5x5-gaussian-linear-1\",\n";
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

wgpu::ShaderModule CreateShaderModule(const wgpu::Device& device, const std::string& wgslSource) {
    wgpu::ShaderSourceWGSL wgslDesc = {};
    wgslDesc.code = wgslSource.c_str();

    wgpu::ShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = &wgslDesc;
    return device.CreateShaderModule(&shaderDesc);
}

void MapBufferBlocking(
    const wgpu::Instance& instance,
    wgpu::Buffer& buffer,
    std::size_t byteSize) {
    struct MapState {
        std::atomic<bool> done{false};
        wgpu::MapAsyncStatus status = wgpu::MapAsyncStatus::Error;
        std::string message;
    };
    MapState mapState;

    buffer.MapAsync(
        wgpu::MapMode::Read,
        0,
        static_cast<std::uint64_t>(byteSize),
        wgpu::CallbackMode::AllowProcessEvents,
        [&mapState](wgpu::MapAsyncStatus status, const char* message) {
            mapState.status = status;
            mapState.message = (message != nullptr) ? std::string(message) : std::string();
            mapState.done.store(true, std::memory_order_release);
        });

    while (!mapState.done.load(std::memory_order_acquire)) {
        instance.ProcessEvents();
        std::this_thread::yield();
    }

    if (mapState.status != wgpu::MapAsyncStatus::Success) {
        std::string message = "readback MapAsync failed";
        if (!mapState.message.empty()) {
            message += ": ";
            message += mapState.message;
        }
        throw std::runtime_error(message);
    }
}

std::vector<std::uint8_t> ReadBufferBlocking(
    const wgpu::Instance& instance,
    wgpu::Buffer& buffer,
    std::size_t byteSize) {
    MapBufferBlocking(instance, buffer, byteSize);
    const void* mapped = buffer.GetConstMappedRange(0, static_cast<std::uint64_t>(byteSize));
    if (mapped == nullptr) {
        throw std::runtime_error("GetConstMappedRange returned null");
    }

    std::vector<std::uint8_t> data(byteSize);
    if (!data.empty()) {
        std::memcpy(data.data(), mapped, byteSize);
    }
    buffer.Unmap();
    return data;
}

void ResolveGpuTimestamps(
    const GpuSession& session,
    const wgpu::CommandEncoder& encoder) {
    if (!session.timestampQueryEnabled) {
        return;
    }
    encoder.ResolveQuerySet(
        session.timestampQuerySet,
        0,
        2,
        session.timestampResolveBuffer,
        0);
    encoder.CopyBufferToBuffer(
        session.timestampResolveBuffer,
        0,
        session.timestampReadbackBuffer,
        0,
        2u * sizeof(std::uint64_t));
}

double ReadGpuTimestampMs(GpuSession& session) {
    if (!session.timestampQueryEnabled) {
        return 0.0;
    }
    constexpr std::size_t kTimestampBytes = 2u * sizeof(std::uint64_t);
    MapBufferBlocking(
        session.instance,
        session.timestampReadbackBuffer,
        kTimestampBytes);
    const void* mapped =
        session.timestampReadbackBuffer.GetConstMappedRange(0, kTimestampBytes);
    if (mapped == nullptr) {
        session.timestampReadbackBuffer.Unmap();
        throw std::runtime_error("timestamp GetConstMappedRange returned null");
    }
    std::array<std::uint64_t, 2> timestamps{};
    std::memcpy(timestamps.data(), mapped, kTimestampBytes);
    session.timestampReadbackBuffer.Unmap();
    if (timestamps[1] < timestamps[0]) {
        throw std::runtime_error("GPU timestamp query returned a negative duration");
    }
    // Dawn converts timestamp-query ticks to nanoseconds unless its internal
    // disable_timestamp_query_conversion toggle is explicitly enabled.
    return static_cast<double>(timestamps[1] - timestamps[0]) / 1'000'000.0;
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
    if (elemCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("input too large for u32 dispatch length");
    }
    const std::size_t expectedCount = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (expectedCount != elemCount) {
        throw std::runtime_error("pixel count mismatch between input buffers and dimensions");
    }

    const std::size_t rgbaBytes = elemCount * sizeof(LinearRgba);
    const std::size_t f32Bytes = elemCount * sizeof(float);

    ScaleOutputs outputs;

    struct ParamsData {
        std::uint32_t len;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t qscale;
    };
    const ParamsData paramsData = {
        .len = static_cast<std::uint32_t>(elemCount),
        .width = width,
        .height = height,
        .qscale = kStage0QScale,
    };
    if (!session.preprocessShader || !session.stage0Shader) {
        throw std::runtime_error("failed to create stage0/preprocess shader module");
    }
    if (!session.preprocessBindGroupLayout || !session.preprocessPipelineLayout || !session.preprocessPipeline) {
        throw std::runtime_error("failed to create preprocess pipeline");
    }
    if (!session.stage0BindGroupLayout || !session.stage0PipelineLayout || !session.stage0Pipeline) {
        throw std::runtime_error("failed to create stage0 compute pipeline");
    }

    std::unique_ptr<Stage0Resources>& resourceSlot =
        readIntermediateStats ? session.debugStage0Resources : session.stage0Resources;
    if (!resourceSlot || resourceSlot->capacity < elemCount) {
        auto resources = std::make_unique<Stage0Resources>();
        resources->capacity = elemCount;
        const std::size_t capacityRgbaBytes = elemCount * sizeof(LinearRgba);
        const std::size_t capacityLabBytes = elemCount * sizeof(float) * 4u;
        const std::size_t capacityF32Bytes = elemCount * sizeof(float);
        const std::size_t capacityStatsF32Bytes =
            readIntermediateStats ? capacityF32Bytes : sizeof(float);

        const auto start_CreateBuffers = std::chrono::steady_clock::now();
        wgpu::BufferDescriptor rgbaStorageDesc = {};
        rgbaStorageDesc.size = static_cast<std::uint64_t>(capacityRgbaBytes);
        rgbaStorageDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        resources->input1Buffer = session.device.CreateBuffer(&rgbaStorageDesc);
        resources->input2Buffer = session.device.CreateBuffer(&rgbaStorageDesc);

        wgpu::BufferDescriptor labStorageDesc = {};
        labStorageDesc.size = static_cast<std::uint64_t>(capacityLabBytes);
        labStorageDesc.usage = wgpu::BufferUsage::Storage;
        resources->lab1Buffer = session.device.CreateBuffer(&labStorageDesc);
        resources->lab2Buffer = session.device.CreateBuffer(&labStorageDesc);

        wgpu::BufferDescriptor f32StorageDesc = {};
        f32StorageDesc.size = static_cast<std::uint64_t>(capacityF32Bytes);
        f32StorageDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
        resources->outSsimBuffer = session.device.CreateBuffer(&f32StorageDesc);

        wgpu::BufferDescriptor statsStorageDesc = f32StorageDesc;
        statsStorageDesc.size = static_cast<std::uint64_t>(capacityStatsF32Bytes);
        resources->outMu1Buffer = session.device.CreateBuffer(&statsStorageDesc);
        resources->outMu2Buffer = session.device.CreateBuffer(&statsStorageDesc);
        resources->outVar1Buffer = session.device.CreateBuffer(&statsStorageDesc);
        resources->outVar2Buffer = session.device.CreateBuffer(&statsStorageDesc);
        resources->outCov12Buffer = session.device.CreateBuffer(&statsStorageDesc);

        wgpu::BufferDescriptor readbackDesc = {};
        readbackDesc.size = static_cast<std::uint64_t>(capacityF32Bytes);
        readbackDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        resources->readbackSsimBuffer = session.device.CreateBuffer(&readbackDesc);
        if (readIntermediateStats) {
            resources->readbackMu1Buffer = session.device.CreateBuffer(&readbackDesc);
            resources->readbackMu2Buffer = session.device.CreateBuffer(&readbackDesc);
            resources->readbackVar1Buffer = session.device.CreateBuffer(&readbackDesc);
            resources->readbackVar2Buffer = session.device.CreateBuffer(&readbackDesc);
            resources->readbackCov12Buffer = session.device.CreateBuffer(&readbackDesc);
        }

        wgpu::BufferDescriptor paramsDesc = {};
        paramsDesc.size = static_cast<std::uint64_t>(sizeof(ParamsData));
        paramsDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        resources->paramsBuffer = session.device.CreateBuffer(&paramsDesc);

        if (!resources->input1Buffer || !resources->input2Buffer || !resources->lab1Buffer ||
            !resources->lab2Buffer || !resources->outSsimBuffer || !resources->outMu1Buffer ||
            !resources->outMu2Buffer || !resources->outVar1Buffer || !resources->outVar2Buffer ||
            !resources->outCov12Buffer || !resources->readbackSsimBuffer || !resources->paramsBuffer) {
            throw std::runtime_error("failed to create reusable stage0 buffers");
        }
        if (readIntermediateStats &&
            (!resources->readbackMu1Buffer || !resources->readbackMu2Buffer ||
             !resources->readbackVar1Buffer || !resources->readbackVar2Buffer ||
             !resources->readbackCov12Buffer)) {
            throw std::runtime_error("failed to create reusable stage0 stats readback buffers");
        }
        const auto finish_CreateBuffers = std::chrono::steady_clock::now();
        outputs.createBuffers_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                finish_CreateBuffers - start_CreateBuffers);

        const auto start_CreateBindGroups = std::chrono::steady_clock::now();
        wgpu::BindGroupEntry preprocessBg1Entries[3] = {};
        preprocessBg1Entries[0].binding = 0;
        preprocessBg1Entries[0].buffer = resources->input1Buffer;
        preprocessBg1Entries[0].size = static_cast<std::uint64_t>(capacityRgbaBytes);
        preprocessBg1Entries[1].binding = 1;
        preprocessBg1Entries[1].buffer = resources->lab1Buffer;
        preprocessBg1Entries[1].size = static_cast<std::uint64_t>(capacityLabBytes);
        preprocessBg1Entries[2].binding = 2;
        preprocessBg1Entries[2].buffer = resources->paramsBuffer;
        preprocessBg1Entries[2].size = static_cast<std::uint64_t>(sizeof(ParamsData));

        wgpu::BindGroupEntry preprocessBg2Entries[3] = {};
        preprocessBg2Entries[0].binding = 0;
        preprocessBg2Entries[0].buffer = resources->input2Buffer;
        preprocessBg2Entries[0].size = static_cast<std::uint64_t>(capacityRgbaBytes);
        preprocessBg2Entries[1].binding = 1;
        preprocessBg2Entries[1].buffer = resources->lab2Buffer;
        preprocessBg2Entries[1].size = static_cast<std::uint64_t>(capacityLabBytes);
        preprocessBg2Entries[2].binding = 2;
        preprocessBg2Entries[2].buffer = resources->paramsBuffer;
        preprocessBg2Entries[2].size = static_cast<std::uint64_t>(sizeof(ParamsData));

        wgpu::BindGroupDescriptor preprocessBg1Desc = {};
        preprocessBg1Desc.layout = session.preprocessBindGroupLayout;
        preprocessBg1Desc.entryCount = 3;
        preprocessBg1Desc.entries = preprocessBg1Entries;
        resources->preprocessBindGroup1 = session.device.CreateBindGroup(&preprocessBg1Desc);
        wgpu::BindGroupDescriptor preprocessBg2Desc = {};
        preprocessBg2Desc.layout = session.preprocessBindGroupLayout;
        preprocessBg2Desc.entryCount = 3;
        preprocessBg2Desc.entries = preprocessBg2Entries;
        resources->preprocessBindGroup2 = session.device.CreateBindGroup(&preprocessBg2Desc);

        wgpu::BindGroupEntry bgEntries[9] = {};
        bgEntries[0].binding = 0;
        bgEntries[0].buffer = resources->lab1Buffer;
        bgEntries[0].size = static_cast<std::uint64_t>(capacityLabBytes);
        bgEntries[1].binding = 1;
        bgEntries[1].buffer = resources->lab2Buffer;
        bgEntries[1].size = static_cast<std::uint64_t>(capacityLabBytes);
        bgEntries[2].binding = 2;
        bgEntries[2].buffer = resources->outSsimBuffer;
        bgEntries[2].size = static_cast<std::uint64_t>(capacityF32Bytes);
        bgEntries[3].binding = 3;
        bgEntries[3].buffer = resources->outMu1Buffer;
        bgEntries[3].size = static_cast<std::uint64_t>(capacityStatsF32Bytes);
        bgEntries[4].binding = 4;
        bgEntries[4].buffer = resources->outMu2Buffer;
        bgEntries[4].size = static_cast<std::uint64_t>(capacityStatsF32Bytes);
        bgEntries[5].binding = 5;
        bgEntries[5].buffer = resources->outVar1Buffer;
        bgEntries[5].size = static_cast<std::uint64_t>(capacityStatsF32Bytes);
        bgEntries[6].binding = 6;
        bgEntries[6].buffer = resources->outVar2Buffer;
        bgEntries[6].size = static_cast<std::uint64_t>(capacityStatsF32Bytes);
        bgEntries[7].binding = 7;
        bgEntries[7].buffer = resources->outCov12Buffer;
        bgEntries[7].size = static_cast<std::uint64_t>(capacityStatsF32Bytes);
        bgEntries[8].binding = 8;
        bgEntries[8].buffer = resources->paramsBuffer;
        bgEntries[8].size = static_cast<std::uint64_t>(sizeof(ParamsData));

        wgpu::BindGroupDescriptor bgDesc = {};
        bgDesc.layout = session.stage0BindGroupLayout;
        bgDesc.entryCount = 9;
        bgDesc.entries = bgEntries;
        resources->stage0BindGroup = session.device.CreateBindGroup(&bgDesc);
        if (!resources->preprocessBindGroup1 || !resources->preprocessBindGroup2 ||
            !resources->stage0BindGroup) {
            throw std::runtime_error("failed to create reusable stage0 bind groups");
        }
        const auto finish_CreateBindGroups = std::chrono::steady_clock::now();
        outputs.createBindGroups_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                finish_CreateBindGroups - start_CreateBindGroups);
        resourceSlot = std::move(resources);
    }

    Stage0Resources& resources = *resourceSlot;
    wgpu::Queue queue = session.device.GetQueue();
    const auto start_WriteInputBuffers = std::chrono::steady_clock::now();
    queue.WriteBuffer(resources.input1Buffer, 0, input1.data(), rgbaBytes);
    queue.WriteBuffer(resources.input2Buffer, 0, input2.data(), rgbaBytes);
    queue.WriteBuffer(resources.paramsBuffer, 0, &paramsData, sizeof(ParamsData));
    const auto finish_WriteInputBuffers = std::chrono::steady_clock::now();
    outputs.writeInputBuffers_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            finish_WriteInputBuffers - start_WriteInputBuffers);

    const auto start_DispatchAndSubmit = std::chrono::steady_clock::now();

    const std::uint32_t wgX = (width + 15u) / 16u;
    const std::uint32_t wgY = (height + 15u) / 16u;

    wgpu::CommandEncoder encoder = session.device.CreateCommandEncoder();
    {
        wgpu::ComputePassDescriptor passDesc = {};
        wgpu::PassTimestampWrites timestampWrites = {};
        if (session.timestampQueryEnabled) {
            timestampWrites.querySet = session.timestampQuerySet;
            timestampWrites.beginningOfPassWriteIndex = 0;
            passDesc.timestampWrites = &timestampWrites;
        }
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&passDesc);
        pass.SetPipeline(session.preprocessPipeline);
        pass.SetBindGroup(0, resources.preprocessBindGroup1);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.SetBindGroup(0, resources.preprocessBindGroup2);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.End();
    }
    {
        wgpu::ComputePassDescriptor passDesc = {};
        wgpu::PassTimestampWrites timestampWrites = {};
        if (session.timestampQueryEnabled) {
            timestampWrites.querySet = session.timestampQuerySet;
            timestampWrites.endOfPassWriteIndex = 1;
            passDesc.timestampWrites = &timestampWrites;
        }
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&passDesc);
        pass.SetPipeline(session.stage0Pipeline);
        pass.SetBindGroup(0, resources.stage0BindGroup);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.End();
    }
    encoder.CopyBufferToBuffer(
        resources.outSsimBuffer,
        0,
        resources.readbackSsimBuffer,
        0,
        static_cast<std::uint64_t>(f32Bytes));
    if (readIntermediateStats) {
        encoder.CopyBufferToBuffer(
            resources.outMu1Buffer, 0, resources.readbackMu1Buffer, 0, static_cast<std::uint64_t>(f32Bytes));
        encoder.CopyBufferToBuffer(
            resources.outMu2Buffer, 0, resources.readbackMu2Buffer, 0, static_cast<std::uint64_t>(f32Bytes));
        encoder.CopyBufferToBuffer(
            resources.outVar1Buffer, 0, resources.readbackVar1Buffer, 0, static_cast<std::uint64_t>(f32Bytes));
        encoder.CopyBufferToBuffer(
            resources.outVar2Buffer, 0, resources.readbackVar2Buffer, 0, static_cast<std::uint64_t>(f32Bytes));
        encoder.CopyBufferToBuffer(
            resources.outCov12Buffer, 0, resources.readbackCov12Buffer, 0, static_cast<std::uint64_t>(f32Bytes));
    }
    ResolveGpuTimestamps(session, encoder);

    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    queue.Submit(1, &commandBuffer);
    const auto finish_DispatchAndSubmit = std::chrono::steady_clock::now();
    outputs.dispatchAndSubmit_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_DispatchAndSubmit - start_DispatchAndSubmit);

    
    outputs.width = width;
    outputs.height = height;
    outputs.elemCount = elemCount;
    const auto start_Readback = std::chrono::steady_clock::now();
    const auto ssimBytes =
        ReadBufferBlocking(session.instance, resources.readbackSsimBuffer, f32Bytes);
    outputs.ssimMap.resize(elemCount);
    std::memcpy(outputs.ssimMap.data(), ssimBytes.data(), f32Bytes);
    if (readIntermediateStats) {
        const auto mu1Bytes =
            ReadBufferBlocking(session.instance, resources.readbackMu1Buffer, f32Bytes);
        const auto mu2Bytes =
            ReadBufferBlocking(session.instance, resources.readbackMu2Buffer, f32Bytes);
        const auto var1Bytes =
            ReadBufferBlocking(session.instance, resources.readbackVar1Buffer, f32Bytes);
        const auto var2Bytes =
            ReadBufferBlocking(session.instance, resources.readbackVar2Buffer, f32Bytes);
        const auto cov12Bytes =
            ReadBufferBlocking(session.instance, resources.readbackCov12Buffer, f32Bytes);
        outputs.mu1.resize(elemCount);
        outputs.mu2.resize(elemCount);
        outputs.var1.resize(elemCount);
        outputs.var2.resize(elemCount);
        outputs.cov12.resize(elemCount);
        std::memcpy(outputs.mu1.data(), mu1Bytes.data(), f32Bytes);
        std::memcpy(outputs.mu2.data(), mu2Bytes.data(), f32Bytes);
        std::memcpy(outputs.var1.data(), var1Bytes.data(), f32Bytes);
        std::memcpy(outputs.var2.data(), var2Bytes.data(), f32Bytes);
        std::memcpy(outputs.cov12.data(), cov12Bytes.data(), f32Bytes);
    }
    outputs.gpuTimestampMs = ReadGpuTimestampMs(session);
    const auto finish_Readback = std::chrono::steady_clock::now();
    outputs.readback_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_Readback - start_Readback);

    const auto start_PostProcess = std::chrono::steady_clock::now();
    // Aggregate f32 SSIM values in f64, matching the reference implementation.
    const double ssimSum = SumF32(outputs.ssimMap.data(), elemCount);
    outputs.meanSsim = ssimSum / static_cast<double>(elemCount);
    const double avg =
        std::pow(std::max(outputs.meanSsim, 0.0), std::pow(0.5, static_cast<double>(scaleLevel)));
    const double devSum =
        SumAbsoluteDeviation(outputs.ssimMap.data(), elemCount, avg);
    outputs.ssimScore = 1.0 - (devSum / static_cast<double>(elemCount));
    const auto finish_PostProcess = std::chrono::steady_clock::now();
    outputs.postProcess_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_PostProcess - start_PostProcess);
    return outputs;
}

std::vector<ScaleOutputs> RunStage0BatchCompute(
    GpuSession& session,
    std::span<const std::uint8_t> input1,
    std::span<const std::uint8_t> input2,
    const std::vector<std::uint32_t>& widths,
    const std::vector<std::uint32_t>& heights) {
    const std::size_t levelCount = widths.size();
    if (levelCount == 0 || heights.size() != levelCount) {
        throw std::runtime_error("invalid batch dimensions");
    }

    constexpr std::size_t kBindingAlignment = 256u;
    const auto alignUp = [](std::size_t value) {
        return (value + kBindingAlignment - 1u) & ~(kBindingAlignment - 1u);
    };

    std::vector<std::size_t> outputOffsets(levelCount);
    std::vector<std::size_t> elemCounts(levelCount);
    std::size_t outputBytesTotal = 0;
    for (std::size_t level = 0; level < levelCount; ++level) {
        const std::size_t elemCount =
            static_cast<std::size_t>(widths[level]) * static_cast<std::size_t>(heights[level]);
        elemCounts[level] = elemCount;
        outputOffsets[level] = outputBytesTotal;
        outputBytesTotal += alignUp(elemCount * sizeof(float));
    }

    const std::size_t baseElemCount = elemCounts.front();
    const std::size_t rgba8Bytes = baseElemCount * 4u;
    if (input1.size() != rgba8Bytes || input2.size() != rgba8Bytes) {
        throw std::runtime_error("batch input dimensions do not match pixel count");
    }
    const std::size_t inputBytes = baseElemCount * sizeof(LinearRgba);
    const std::size_t downsampleBytes =
        ((levelCount > 1u) ? elemCounts[1] : 1u) * sizeof(LinearRgba);
    const std::size_t baseLabBytes = baseElemCount * sizeof(float) * 4u;
    const std::size_t baseF32Bytes = baseElemCount * sizeof(float);
    const std::size_t paramsBytes = levelCount * kBindingAlignment;

    std::vector<ScaleOutputs> outputs(levelCount);
    if (!session.batchStage0Resources ||
        session.batchStage0Resources->rgba8CapacityBytes < rgba8Bytes ||
        session.batchStage0Resources->inputCapacityBytes < inputBytes ||
        session.batchStage0Resources->downsampleCapacityBytes < downsampleBytes ||
        session.batchStage0Resources->outputCapacityBytes < outputBytesTotal ||
        session.batchStage0Resources->baseCapacity < baseElemCount) {
        auto resources = std::make_unique<BatchStage0Resources>();
        resources->rgba8CapacityBytes = rgba8Bytes;
        resources->inputCapacityBytes = inputBytes;
        resources->downsampleCapacityBytes = downsampleBytes;
        resources->outputCapacityBytes = outputBytesTotal;
        resources->baseCapacity = baseElemCount;

        const auto startCreateBuffers = std::chrono::steady_clock::now();
        wgpu::BufferDescriptor rgba8Desc = {};
        rgba8Desc.size = static_cast<std::uint64_t>(rgba8Bytes);
        rgba8Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        resources->rgba8Input1Buffer = session.device.CreateBuffer(&rgba8Desc);
        resources->rgba8Input2Buffer = session.device.CreateBuffer(&rgba8Desc);

        wgpu::BufferDescriptor inputDesc = {};
        inputDesc.size = static_cast<std::uint64_t>(inputBytes);
        inputDesc.usage = wgpu::BufferUsage::Storage;
        resources->input1Buffer = session.device.CreateBuffer(&inputDesc);
        resources->input2Buffer = session.device.CreateBuffer(&inputDesc);

        wgpu::BufferDescriptor downsampleDesc = {};
        downsampleDesc.size = static_cast<std::uint64_t>(downsampleBytes);
        downsampleDesc.usage = wgpu::BufferUsage::Storage;
        resources->downsample1Buffer = session.device.CreateBuffer(&downsampleDesc);
        resources->downsample2Buffer = session.device.CreateBuffer(&downsampleDesc);

        wgpu::BufferDescriptor labDesc = {};
        labDesc.size = static_cast<std::uint64_t>(baseLabBytes);
        labDesc.usage = wgpu::BufferUsage::Storage;
        resources->lab1Buffer = session.device.CreateBuffer(&labDesc);
        resources->lab2Buffer = session.device.CreateBuffer(&labDesc);

        wgpu::BufferDescriptor ssimDesc = {};
        ssimDesc.size = static_cast<std::uint64_t>(baseF32Bytes);
        ssimDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
        resources->outSsimBuffer = session.device.CreateBuffer(&ssimDesc);

        wgpu::BufferDescriptor statsDesc = ssimDesc;
        statsDesc.size = sizeof(float);
        resources->outMu1Buffer = session.device.CreateBuffer(&statsDesc);
        resources->outMu2Buffer = session.device.CreateBuffer(&statsDesc);
        resources->outVar1Buffer = session.device.CreateBuffer(&statsDesc);
        resources->outVar2Buffer = session.device.CreateBuffer(&statsDesc);
        resources->outCov12Buffer = session.device.CreateBuffer(&statsDesc);

        wgpu::BufferDescriptor readbackDesc = {};
        readbackDesc.size = static_cast<std::uint64_t>(outputBytesTotal);
        readbackDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        resources->readbackSsimBuffer = session.device.CreateBuffer(&readbackDesc);

        wgpu::BufferDescriptor paramsDesc = {};
        paramsDesc.size = static_cast<std::uint64_t>(paramsBytes);
        paramsDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        resources->paramsBuffer = session.device.CreateBuffer(&paramsDesc);
        resources->downsampleParamsBuffer = session.device.CreateBuffer(&paramsDesc);

        wgpu::BufferDescriptor lutDesc = {};
        lutDesc.size = 256u * sizeof(float);
        lutDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        resources->srgbToLinearLutBuffer = session.device.CreateBuffer(&lutDesc);

        if (!resources->rgba8Input1Buffer || !resources->rgba8Input2Buffer ||
            !resources->input1Buffer || !resources->input2Buffer ||
            !resources->downsample1Buffer || !resources->downsample2Buffer ||
            !resources->lab1Buffer || !resources->lab2Buffer ||
            !resources->outSsimBuffer || !resources->outMu1Buffer ||
            !resources->outMu2Buffer || !resources->outVar1Buffer ||
            !resources->outVar2Buffer || !resources->outCov12Buffer ||
            !resources->readbackSsimBuffer || !resources->paramsBuffer ||
            !resources->downsampleParamsBuffer ||
            !resources->srgbToLinearLutBuffer) {
            throw std::runtime_error("failed to create batch stage0 buffers");
        }
        const auto finishCreateBuffers = std::chrono::steady_clock::now();
        outputs.front().createBuffers_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                finishCreateBuffers - startCreateBuffers);
        session.batchStage0Resources = std::move(resources);
    }

    BatchStage0Resources& resources = *session.batchStage0Resources;
    std::vector<wgpu::BindGroup> preprocessBindGroups1(levelCount);
    std::vector<wgpu::BindGroup> preprocessBindGroups2(levelCount);
    std::vector<wgpu::BindGroup> stage0BindGroups(levelCount);
    std::vector<wgpu::BindGroup> downsampleBindGroups1(levelCount - 1u);
    std::vector<wgpu::BindGroup> downsampleBindGroups2(levelCount - 1u);

    const auto startCreateBindGroups = std::chrono::steady_clock::now();
    wgpu::BindGroupEntry convertEntries1[4] = {};
    convertEntries1[0].binding = 0;
    convertEntries1[0].buffer = resources.rgba8Input1Buffer;
    convertEntries1[0].size = static_cast<std::uint64_t>(rgba8Bytes);
    convertEntries1[1].binding = 1;
    convertEntries1[1].buffer = resources.input1Buffer;
    convertEntries1[1].size = static_cast<std::uint64_t>(inputBytes);
    convertEntries1[2].binding = 2;
    convertEntries1[2].buffer = resources.srgbToLinearLutBuffer;
    convertEntries1[2].size = 256u * sizeof(float);
    convertEntries1[3].binding = 3;
    convertEntries1[3].buffer = resources.paramsBuffer;
    convertEntries1[3].size = 4u * sizeof(std::uint32_t);
    wgpu::BindGroupDescriptor convertDesc1 = {};
    convertDesc1.layout = session.rgba8ToLinearBindGroupLayout;
    convertDesc1.entryCount = 4;
    convertDesc1.entries = convertEntries1;
    wgpu::BindGroup convertBindGroup1 =
        session.device.CreateBindGroup(&convertDesc1);

    wgpu::BindGroupEntry convertEntries2[4] = {};
    std::copy(
        std::begin(convertEntries1),
        std::end(convertEntries1),
        std::begin(convertEntries2));
    convertEntries2[0].buffer = resources.rgba8Input2Buffer;
    convertEntries2[1].buffer = resources.input2Buffer;
    wgpu::BindGroupDescriptor convertDesc2 = convertDesc1;
    convertDesc2.entries = convertEntries2;
    wgpu::BindGroup convertBindGroup2 =
        session.device.CreateBindGroup(&convertDesc2);
    if (!convertBindGroup1 || !convertBindGroup2) {
        throw std::runtime_error("failed to create rgba8 conversion bind groups");
    }

    for (std::size_t level = 0; level < levelCount; ++level) {
        const std::uint64_t rgbaBytes =
            static_cast<std::uint64_t>(elemCounts[level] * sizeof(LinearRgba));
        const std::uint64_t f32Bytes =
            static_cast<std::uint64_t>(elemCounts[level] * sizeof(float));
        const std::uint64_t paramsOffset =
            static_cast<std::uint64_t>(level * kBindingAlignment);
        const wgpu::Buffer& currentInput1 =
            ((level & 1u) == 0u) ? resources.input1Buffer : resources.downsample1Buffer;
        const wgpu::Buffer& currentInput2 =
            ((level & 1u) == 0u) ? resources.input2Buffer : resources.downsample2Buffer;

        wgpu::BindGroupEntry preprocessEntries1[3] = {};
        preprocessEntries1[0].binding = 0;
        preprocessEntries1[0].buffer = currentInput1;
        preprocessEntries1[0].size = rgbaBytes;
        preprocessEntries1[1].binding = 1;
        preprocessEntries1[1].buffer = resources.lab1Buffer;
        preprocessEntries1[1].size = rgbaBytes;
        preprocessEntries1[2].binding = 2;
        preprocessEntries1[2].buffer = resources.paramsBuffer;
        preprocessEntries1[2].offset = paramsOffset;
        preprocessEntries1[2].size = 4u * sizeof(std::uint32_t);

        wgpu::BindGroupDescriptor preprocessDesc1 = {};
        preprocessDesc1.layout = session.preprocessBindGroupLayout;
        preprocessDesc1.entryCount = 3;
        preprocessDesc1.entries = preprocessEntries1;
        preprocessBindGroups1[level] = session.device.CreateBindGroup(&preprocessDesc1);

        wgpu::BindGroupEntry preprocessEntries2[3] = {};
        std::copy(
            std::begin(preprocessEntries1),
            std::end(preprocessEntries1),
            std::begin(preprocessEntries2));
        preprocessEntries2[0].buffer = currentInput2;
        preprocessEntries2[1].buffer = resources.lab2Buffer;
        wgpu::BindGroupDescriptor preprocessDesc2 = preprocessDesc1;
        preprocessDesc2.entries = preprocessEntries2;
        preprocessBindGroups2[level] = session.device.CreateBindGroup(&preprocessDesc2);

        wgpu::BindGroupEntry stageEntries[4] = {};
        stageEntries[0].binding = 0;
        stageEntries[0].buffer = resources.lab1Buffer;
        stageEntries[0].size = rgbaBytes;
        stageEntries[1].binding = 1;
        stageEntries[1].buffer = resources.lab2Buffer;
        stageEntries[1].size = rgbaBytes;
        stageEntries[2].binding = 2;
        stageEntries[2].buffer = resources.outSsimBuffer;
        stageEntries[2].size = f32Bytes;
        stageEntries[3].binding = 3;
        stageEntries[3].buffer = resources.paramsBuffer;
        stageEntries[3].offset = paramsOffset;
        stageEntries[3].size = 4u * sizeof(std::uint32_t);

        wgpu::BindGroupDescriptor stageDesc = {};
        stageDesc.layout = session.stage0ScoreBindGroupLayout;
        stageDesc.entryCount = 4;
        stageDesc.entries = stageEntries;
        stage0BindGroups[level] = session.device.CreateBindGroup(&stageDesc);

        if (!preprocessBindGroups1[level] || !preprocessBindGroups2[level] ||
            !stage0BindGroups[level]) {
            throw std::runtime_error("failed to create batch stage0 bind groups");
        }

        if (level + 1u < levelCount) {
            const std::uint64_t nextRgbaBytes = static_cast<std::uint64_t>(
                elemCounts[level + 1u] * sizeof(LinearRgba));
            const wgpu::Buffer& nextInput1 =
                ((level & 1u) == 0u) ? resources.downsample1Buffer : resources.input1Buffer;
            const wgpu::Buffer& nextInput2 =
                ((level & 1u) == 0u) ? resources.downsample2Buffer : resources.input2Buffer;

            wgpu::BindGroupEntry downsampleEntries1[3] = {};
            downsampleEntries1[0].binding = 0;
            downsampleEntries1[0].buffer = currentInput1;
            downsampleEntries1[0].size = rgbaBytes;
            downsampleEntries1[1].binding = 1;
            downsampleEntries1[1].buffer = nextInput1;
            downsampleEntries1[1].size = nextRgbaBytes;
            downsampleEntries1[2].binding = 2;
            downsampleEntries1[2].buffer = resources.downsampleParamsBuffer;
            downsampleEntries1[2].offset = paramsOffset;
            downsampleEntries1[2].size = 4u * sizeof(std::uint32_t);

            wgpu::BindGroupDescriptor downsampleDesc1 = {};
            downsampleDesc1.layout = session.downsampleBindGroupLayout;
            downsampleDesc1.entryCount = 3;
            downsampleDesc1.entries = downsampleEntries1;
            downsampleBindGroups1[level] =
                session.device.CreateBindGroup(&downsampleDesc1);

            wgpu::BindGroupEntry downsampleEntries2[3] = {};
            std::copy(
                std::begin(downsampleEntries1),
                std::end(downsampleEntries1),
                std::begin(downsampleEntries2));
            downsampleEntries2[0].buffer = currentInput2;
            downsampleEntries2[1].buffer = nextInput2;
            wgpu::BindGroupDescriptor downsampleDesc2 = downsampleDesc1;
            downsampleDesc2.entries = downsampleEntries2;
            downsampleBindGroups2[level] =
                session.device.CreateBindGroup(&downsampleDesc2);

            if (!downsampleBindGroups1[level] || !downsampleBindGroups2[level]) {
                throw std::runtime_error("failed to create batch downsample bind groups");
            }
        }
    }
    const auto finishCreateBindGroups = std::chrono::steady_clock::now();
    outputs.front().createBindGroups_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            finishCreateBindGroups - startCreateBindGroups);

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

    wgpu::Queue queue = session.device.GetQueue();
    const auto startWriteBuffers = std::chrono::steady_clock::now();
    queue.WriteBuffer(resources.rgba8Input1Buffer, 0, input1.data(), rgba8Bytes);
    queue.WriteBuffer(resources.rgba8Input2Buffer, 0, input2.data(), rgba8Bytes);
    const auto& srgbToLinearLut = SrgbToLinearLut();
    queue.WriteBuffer(
        resources.srgbToLinearLutBuffer,
        0,
        srgbToLinearLut.data(),
        srgbToLinearLut.size() * sizeof(float));
    for (std::size_t level = 0; level < levelCount; ++level) {
        const ParamsData params = {
            .len = static_cast<std::uint32_t>(elemCounts[level]),
            .width = widths[level],
            .height = heights[level],
            .qscale = kStage0QScale,
        };
        queue.WriteBuffer(
            resources.paramsBuffer,
            static_cast<std::uint64_t>(level * kBindingAlignment),
            &params,
            sizeof(params));
        if (level + 1u < levelCount) {
            const DownsampleParamsData downsampleParams = {
                .inWidth = widths[level],
                .inHeight = heights[level],
                .outWidth = widths[level + 1u],
                .outHeight = heights[level + 1u],
            };
            queue.WriteBuffer(
                resources.downsampleParamsBuffer,
                static_cast<std::uint64_t>(level * kBindingAlignment),
                &downsampleParams,
                sizeof(downsampleParams));
        }
    }
    const auto finishWriteBuffers = std::chrono::steady_clock::now();
    outputs.front().writeInputBuffers_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            finishWriteBuffers - startWriteBuffers);

    const auto startDispatch = std::chrono::steady_clock::now();
    wgpu::CommandEncoder encoder = session.device.CreateCommandEncoder();
    {
        const std::uint32_t wgX = (widths.front() + 15u) / 16u;
        const std::uint32_t wgY = (heights.front() + 15u) / 16u;
        wgpu::ComputePassDescriptor passDesc = {};
        wgpu::PassTimestampWrites timestampWrites = {};
        if (session.timestampQueryEnabled) {
            timestampWrites.querySet = session.timestampQuerySet;
            timestampWrites.beginningOfPassWriteIndex = 0;
            passDesc.timestampWrites = &timestampWrites;
        }
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&passDesc);
        pass.SetPipeline(session.rgba8ToLinearPipeline);
        pass.SetBindGroup(0, convertBindGroup1);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.SetBindGroup(0, convertBindGroup2);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.End();
    }
    for (std::size_t level = 0; level < levelCount; ++level) {
        const std::uint32_t wgX = (widths[level] + 15u) / 16u;
        const std::uint32_t wgY = (heights[level] + 15u) / 16u;
        wgpu::ComputePassDescriptor passDesc = {};
        wgpu::PassTimestampWrites timestampWrites = {};
        if (session.timestampQueryEnabled && level + 1u == levelCount) {
            timestampWrites.querySet = session.timestampQuerySet;
            timestampWrites.endOfPassWriteIndex = 1;
            passDesc.timestampWrites = &timestampWrites;
        }
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&passDesc);
        pass.SetPipeline(session.preprocessPipeline);
        pass.SetBindGroup(0, preprocessBindGroups1[level]);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.SetBindGroup(0, preprocessBindGroups2[level]);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.SetPipeline(session.stage0ScorePipeline);
        pass.SetBindGroup(0, stage0BindGroups[level]);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.End();

        encoder.CopyBufferToBuffer(
            resources.outSsimBuffer,
            0,
            resources.readbackSsimBuffer,
            static_cast<std::uint64_t>(outputOffsets[level]),
            static_cast<std::uint64_t>(elemCounts[level] * sizeof(float)));

        if (level + 1u < levelCount) {
            const std::uint32_t downsampleWgX = (widths[level + 1u] + 15u) / 16u;
            const std::uint32_t downsampleWgY = (heights[level + 1u] + 15u) / 16u;
            wgpu::ComputePassDescriptor downsamplePassDesc = {};
            wgpu::ComputePassEncoder downsamplePass =
                encoder.BeginComputePass(&downsamplePassDesc);
            downsamplePass.SetPipeline(session.downsamplePipeline);
            downsamplePass.SetBindGroup(0, downsampleBindGroups1[level]);
            downsamplePass.DispatchWorkgroups(downsampleWgX, downsampleWgY, 1);
            downsamplePass.SetBindGroup(0, downsampleBindGroups2[level]);
            downsamplePass.DispatchWorkgroups(downsampleWgX, downsampleWgY, 1);
            downsamplePass.End();
        }
    }
    ResolveGpuTimestamps(session, encoder);
    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    queue.Submit(1, &commandBuffer);
    const auto finishDispatch = std::chrono::steady_clock::now();
    outputs.front().dispatchAndSubmit_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            finishDispatch - startDispatch);

    const auto startReadback = std::chrono::steady_clock::now();
    MapBufferBlocking(
        session.instance,
        resources.readbackSsimBuffer,
        outputBytesTotal);
    const auto* ssimBytes = static_cast<const std::uint8_t*>(
        resources.readbackSsimBuffer.GetConstMappedRange(
            0,
            static_cast<std::uint64_t>(outputBytesTotal)));
    if (ssimBytes == nullptr) {
        resources.readbackSsimBuffer.Unmap();
        throw std::runtime_error("GetConstMappedRange returned null");
    }
    outputs.front().gpuTimestampMs = ReadGpuTimestampMs(session);
    const auto finishReadback = std::chrono::steady_clock::now();
    outputs.front().readback_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            finishReadback - startReadback);

    const auto startPostProcess = std::chrono::steady_clock::now();
    const auto processLevel = [&](std::size_t level) {
        ScaleOutputs& output = outputs[level];
        output.width = widths[level];
        output.height = heights[level];
        output.elemCount = elemCounts[level];
        const float* ssimValues = reinterpret_cast<const float*>(
            ssimBytes + outputOffsets[level]);

        const double ssimSum = SumF32(ssimValues, elemCounts[level]);
        output.meanSsim = ssimSum / static_cast<double>(elemCounts[level]);
        const double avg = std::pow(
            std::max(output.meanSsim, 0.0),
            std::pow(0.5, static_cast<double>(level)));
        const double devSum =
            SumAbsoluteDeviation(ssimValues, elemCounts[level], avg);
        output.ssimScore = 1.0 - (devSum / static_cast<double>(elemCounts[level]));
    };
    if (levelCount > 1u && baseElemCount >= 65536u) {
        auto remainingLevels = std::async(std::launch::async, [&] {
            for (std::size_t level = 1; level < levelCount; ++level) {
                processLevel(level);
            }
        });
        processLevel(0);
        remainingLevels.get();
    } else {
        for (std::size_t level = 0; level < levelCount; ++level) {
            processLevel(level);
        }
    }
    resources.readbackSsimBuffer.Unmap();
    const auto finishPostProcess = std::chrono::steady_clock::now();
    outputs.front().postProcess_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            finishPostProcess - startPostProcess);
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

DownsampleOutputs RunDownsample2x2Compute(
    const GpuSession& session,
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
    const std::size_t outCount = static_cast<std::size_t>(outWidth) * static_cast<std::size_t>(outHeight);

    const std::size_t inBytes = inCount * sizeof(LinearRgba);
    const std::size_t outBytes = outCount * sizeof(LinearRgba);

    struct ParamsData {
        std::uint32_t inWidth;
        std::uint32_t inHeight;
        std::uint32_t outWidth;
        std::uint32_t outHeight;
    };
    const ParamsData paramsData = {
        .inWidth = inWidth,
        .inHeight = inHeight,
        .outWidth = outWidth,
        .outHeight = outHeight,
    };
    DownsampleOutputs out;
    const auto start_CreateBuffers = std::chrono::steady_clock::now();

    wgpu::BufferDescriptor inDesc = {};
    inDesc.size = static_cast<std::uint64_t>(inBytes);
    inDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    inDesc.mappedAtCreation = false;
    wgpu::Buffer inBuffer = session.device.CreateBuffer(&inDesc);

    wgpu::BufferDescriptor outDesc = {};
    outDesc.size = static_cast<std::uint64_t>(outBytes);
    outDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    outDesc.mappedAtCreation = false;
    wgpu::Buffer outBuffer = session.device.CreateBuffer(&outDesc);

    wgpu::BufferDescriptor readbackDesc = {};
    readbackDesc.size = static_cast<std::uint64_t>(outBytes);
    readbackDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    readbackDesc.mappedAtCreation = false;
    wgpu::Buffer readbackBuffer = session.device.CreateBuffer(&readbackDesc);

    wgpu::BufferDescriptor paramsDesc = {};
    paramsDesc.size = static_cast<std::uint64_t>(sizeof(ParamsData));
    paramsDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    paramsDesc.mappedAtCreation = false;
    wgpu::Buffer paramsBuffer = session.device.CreateBuffer(&paramsDesc);

    if (!inBuffer || !outBuffer || !readbackBuffer || !paramsBuffer) {
        throw std::runtime_error("failed to create downsample buffers");
    }
    const auto finish_CreateBuffers = std::chrono::steady_clock::now();
    out.createBuffers_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_CreateBuffers - start_CreateBuffers);

    wgpu::Queue queue = session.device.GetQueue();
    const auto start_WriteInputBuffers = std::chrono::steady_clock::now();
    queue.WriteBuffer(inBuffer, 0, input.data(), inBytes);
    queue.WriteBuffer(paramsBuffer, 0, &paramsData, sizeof(ParamsData));
    const auto finish_WriteInputBuffers = std::chrono::steady_clock::now();
    out.writeInputBuffers_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_WriteInputBuffers - start_WriteInputBuffers);
    if (!session.downsampleShader) {
        throw std::runtime_error("failed to create downsample shader module");
    }
    if (!session.downsampleBindGroupLayout || !session.downsamplePipelineLayout || !session.downsamplePipeline) {
        throw std::runtime_error("failed to create downsample pipeline");
    }
    const auto start_CreateBindGroups = std::chrono::steady_clock::now();

    wgpu::BindGroupEntry bgEntries[3] = {};
    bgEntries[0].binding = 0;
    bgEntries[0].buffer = inBuffer;
    bgEntries[0].size = static_cast<std::uint64_t>(inBytes);
    bgEntries[1].binding = 1;
    bgEntries[1].buffer = outBuffer;
    bgEntries[1].size = static_cast<std::uint64_t>(outBytes);
    bgEntries[2].binding = 2;
    bgEntries[2].buffer = paramsBuffer;
    bgEntries[2].size = static_cast<std::uint64_t>(sizeof(ParamsData));

    wgpu::BindGroupDescriptor bgDesc = {};
    bgDesc.layout = session.downsampleBindGroupLayout;
    bgDesc.entryCount = 3;
    bgDesc.entries = bgEntries;
    wgpu::BindGroup bindGroup = session.device.CreateBindGroup(&bgDesc);
    if (!bindGroup) {
        throw std::runtime_error("failed to create downsample bind group");
    }
    const auto finish_CreateBindGroups = std::chrono::steady_clock::now();
    out.createBindGroups_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_CreateBindGroups - start_CreateBindGroups);
    const auto start_DispatchAndSubmit = std::chrono::steady_clock::now();

    wgpu::CommandEncoder encoder = session.device.CreateCommandEncoder();
    {
        wgpu::ComputePassDescriptor passDesc = {};
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&passDesc);
        pass.SetPipeline(session.downsamplePipeline);
        pass.SetBindGroup(0, bindGroup);
        const std::uint32_t dsWgX = (outWidth + 15u) / 16u;
        const std::uint32_t dsWgY = (outHeight + 15u) / 16u;
        pass.DispatchWorkgroups(dsWgX, dsWgY, 1);
        pass.End();
    }
    encoder.CopyBufferToBuffer(outBuffer, 0, readbackBuffer, 0, static_cast<std::uint64_t>(outBytes));
    wgpu::CommandBuffer cb = encoder.Finish();
    queue.Submit(1, &cb);
    const auto finish_DispatchAndSubmit = std::chrono::steady_clock::now();
    out.dispatchAndSubmit_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_DispatchAndSubmit - start_DispatchAndSubmit);

    const auto start_Readback = std::chrono::steady_clock::now();
    const auto outBytesVec = ReadBufferBlocking(session.instance, readbackBuffer, outBytes);
    out.width = outWidth;
    out.height = outHeight;
    out.pixels.resize(outCount);
    std::memcpy(out.pixels.data(), outBytesVec.data(), outBytes);
    const auto finish_Readback = std::chrono::steady_clock::now();
    out.readback_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_Readback - start_Readback);
    return out;
}

wgpu::Adapter RequestAdapterBlocking(const wgpu::Instance& instance) {
    struct RequestState {
        std::atomic<bool> done{false};
        wgpu::RequestAdapterStatus status = wgpu::RequestAdapterStatus::Error;
        wgpu::Adapter adapter = nullptr;
        std::string message;
    };
    RequestState state;

    wgpu::RequestAdapterOptions options = {};
#if defined(_WIN32)
    options.backendType = wgpu::BackendType::D3D12;
#endif
    instance.RequestAdapter(
        &options,
        wgpu::CallbackMode::AllowProcessEvents,
        [&state](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char* message) {
            state.status = status;
            state.adapter = adapter;
            state.message = (message != nullptr) ? std::string(message) : std::string();
            state.done.store(true, std::memory_order_release);
        });

    while (!state.done.load(std::memory_order_acquire)) {
        instance.ProcessEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (state.status != wgpu::RequestAdapterStatus::Success || !state.adapter) {
        std::string message = "failed to request adapter";
        if (!state.message.empty()) {
            message += ": ";
            message += state.message;
        }
        throw std::runtime_error(message);
    }
    return state.adapter;
}

wgpu::Device RequestDeviceBlocking(
    const wgpu::Instance& instance,
    const wgpu::Adapter& adapter,
    bool enableTimestampQueries) {
    struct RequestState {
        std::atomic<bool> done{false};
        wgpu::RequestDeviceStatus status = wgpu::RequestDeviceStatus::Error;
        wgpu::Device device = nullptr;
        std::string message;
    };
    RequestState state;

    wgpu::DeviceDescriptor descriptor = {};
    const wgpu::FeatureName timestampFeature = wgpu::FeatureName::TimestampQuery;
    if (enableTimestampQueries) {
        if (!adapter.HasFeature(timestampFeature)) {
            throw std::runtime_error(
                "selected WebGPU adapter does not support TimestampQuery");
        }
        descriptor.requiredFeatureCount = 1;
        descriptor.requiredFeatures = &timestampFeature;
    }
    adapter.RequestDevice(
        &descriptor,
        wgpu::CallbackMode::AllowProcessEvents,
        [&state](wgpu::RequestDeviceStatus status, wgpu::Device device, const char* message) {
            state.status = status;
            state.device = device;
            state.message = (message != nullptr) ? std::string(message) : std::string();
            state.done.store(true, std::memory_order_release);
        });

    while (!state.done.load(std::memory_order_acquire)) {
        instance.ProcessEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (state.status != wgpu::RequestDeviceStatus::Success || !state.device) {
        std::string message = "failed to request device";
        if (!state.message.empty()) {
            message += ": ";
            message += state.message;
        }
        throw std::runtime_error(message);
    }
    return state.device;
}

GpuSession CreateGpuSession(
    const std::string& labPreprocessShaderSource,
    const std::string& stage0ShaderSource,
    const std::string& stage0ScoreShaderSource,
    const std::string& downsampleShaderSource,
    const std::string& rgba8ToLinearShaderSource,
    bool enableDebugPipeline,
    bool enableTimestampQueries) {
    GpuSession session;

    dawnProcSetProcs(&dawn::native::GetProcs());

    session.instance = wgpu::CreateInstance();
    if (!session.instance) {
        throw std::runtime_error("failed to create WGPU instance");
    }

    session.adapter = RequestAdapterBlocking(session.instance);
    session.device =
        RequestDeviceBlocking(session.instance, session.adapter, enableTimestampQueries);
    session.timestampQueryEnabled = enableTimestampQueries;

    if (session.timestampQueryEnabled) {
        const auto startCreateTimestampResources = std::chrono::steady_clock::now();
        wgpu::QuerySetDescriptor querySetDescriptor = {};
        querySetDescriptor.type = wgpu::QueryType::Timestamp;
        querySetDescriptor.count = 2;
        session.timestampQuerySet =
            session.device.CreateQuerySet(&querySetDescriptor);

        wgpu::BufferDescriptor resolveDescriptor = {};
        resolveDescriptor.size = 2u * sizeof(std::uint64_t);
        resolveDescriptor.usage =
            wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc;
        session.timestampResolveBuffer =
            session.device.CreateBuffer(&resolveDescriptor);

        wgpu::BufferDescriptor readbackDescriptor = {};
        readbackDescriptor.size = 2u * sizeof(std::uint64_t);
        readbackDescriptor.usage =
            wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        session.timestampReadbackBuffer =
            session.device.CreateBuffer(&readbackDescriptor);
        if (!session.timestampQuerySet || !session.timestampResolveBuffer ||
            !session.timestampReadbackBuffer) {
            throw std::runtime_error(
                "failed to create TimestampQuery profiling resources");
        }
        const auto finishCreateTimestampResources = std::chrono::steady_clock::now();
        session.initProfiling.createBuffersTime =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                finishCreateTimestampResources - startCreateTimestampResources);
    }

    wgpu::AdapterInfo adapterInfo;
    if (session.adapter.GetInfo(&adapterInfo)) {
        const std::string_view description = static_cast<std::string_view>(adapterInfo.description);
        const std::string_view deviceName = static_cast<std::string_view>(adapterInfo.device);
        if (!description.empty()) {
            session.adapterName = std::string(description);
        } else if (!deviceName.empty()) {
            session.adapterName = std::string(deviceName);
        }
    }

    const auto startCreateShaderModule = std::chrono::steady_clock::now();
    session.preprocessShader = CreateShaderModule(session.device, labPreprocessShaderSource);
    if (enableDebugPipeline) {
        session.stage0Shader = CreateShaderModule(session.device, stage0ShaderSource);
    } else {
        session.stage0ScoreShader =
            CreateShaderModule(session.device, stage0ScoreShaderSource);
    }
    session.downsampleShader = CreateShaderModule(session.device, downsampleShaderSource);
    session.rgba8ToLinearShader =
        CreateShaderModule(session.device, rgba8ToLinearShaderSource);
    const auto finishCreateShaderModule = std::chrono::steady_clock::now();
    session.initProfiling.createShaderModuleTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(finishCreateShaderModule - startCreateShaderModule);
    if (!session.preprocessShader ||
        (enableDebugPipeline ? !session.stage0Shader : !session.stage0ScoreShader) ||
        !session.downsampleShader ||
        !session.rgba8ToLinearShader) {
        throw std::runtime_error("failed to create reusable shader modules");
    }

    struct Stage0ParamsData {
        std::uint32_t len;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t qscale;
    };
    const auto startCreatePipelineLayouts = std::chrono::steady_clock::now();

    wgpu::BindGroupLayoutEntry rgba8ToLinearLayoutEntries[4] = {};
    rgba8ToLinearLayoutEntries[0].binding = 0;
    rgba8ToLinearLayoutEntries[0].visibility = wgpu::ShaderStage::Compute;
    rgba8ToLinearLayoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    rgba8ToLinearLayoutEntries[1].binding = 1;
    rgba8ToLinearLayoutEntries[1].visibility = wgpu::ShaderStage::Compute;
    rgba8ToLinearLayoutEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
    rgba8ToLinearLayoutEntries[2].binding = 2;
    rgba8ToLinearLayoutEntries[2].visibility = wgpu::ShaderStage::Compute;
    rgba8ToLinearLayoutEntries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    rgba8ToLinearLayoutEntries[2].buffer.minBindingSize = 256u * sizeof(float);
    rgba8ToLinearLayoutEntries[3].binding = 3;
    rgba8ToLinearLayoutEntries[3].visibility = wgpu::ShaderStage::Compute;
    rgba8ToLinearLayoutEntries[3].buffer.type = wgpu::BufferBindingType::Uniform;
    rgba8ToLinearLayoutEntries[3].buffer.minBindingSize = sizeof(Stage0ParamsData);
    wgpu::BindGroupLayoutDescriptor rgba8ToLinearBglDesc = {};
    rgba8ToLinearBglDesc.entryCount = 4;
    rgba8ToLinearBglDesc.entries = rgba8ToLinearLayoutEntries;
    session.rgba8ToLinearBindGroupLayout =
        session.device.CreateBindGroupLayout(&rgba8ToLinearBglDesc);

    wgpu::PipelineLayoutDescriptor rgba8ToLinearPlDesc = {};
    rgba8ToLinearPlDesc.bindGroupLayoutCount = 1;
    rgba8ToLinearPlDesc.bindGroupLayouts = &session.rgba8ToLinearBindGroupLayout;
    session.rgba8ToLinearPipelineLayout =
        session.device.CreatePipelineLayout(&rgba8ToLinearPlDesc);

    wgpu::BindGroupLayoutEntry preprocessLayoutEntries[3] = {};
    preprocessLayoutEntries[0].binding = 0;
    preprocessLayoutEntries[0].visibility = wgpu::ShaderStage::Compute;
    preprocessLayoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    preprocessLayoutEntries[1].binding = 1;
    preprocessLayoutEntries[1].visibility = wgpu::ShaderStage::Compute;
    preprocessLayoutEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
    preprocessLayoutEntries[2].binding = 2;
    preprocessLayoutEntries[2].visibility = wgpu::ShaderStage::Compute;
    preprocessLayoutEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
    preprocessLayoutEntries[2].buffer.minBindingSize = sizeof(Stage0ParamsData);
    wgpu::BindGroupLayoutDescriptor preprocessBglDesc = {};
    preprocessBglDesc.entryCount = 3;
    preprocessBglDesc.entries = preprocessLayoutEntries;
    session.preprocessBindGroupLayout = session.device.CreateBindGroupLayout(&preprocessBglDesc);

    wgpu::PipelineLayoutDescriptor preprocessPlDesc = {};
    preprocessPlDesc.bindGroupLayoutCount = 1;
    preprocessPlDesc.bindGroupLayouts = &session.preprocessBindGroupLayout;
    session.preprocessPipelineLayout = session.device.CreatePipelineLayout(&preprocessPlDesc);

    wgpu::BindGroupLayoutEntry stage0LayoutEntries[9] = {};
    for (std::uint32_t i = 0; i < 8; ++i) {
        stage0LayoutEntries[i].binding = i;
        stage0LayoutEntries[i].visibility = wgpu::ShaderStage::Compute;
        stage0LayoutEntries[i].buffer.type =
            (i <= 1) ? wgpu::BufferBindingType::ReadOnlyStorage : wgpu::BufferBindingType::Storage;
        stage0LayoutEntries[i].buffer.minBindingSize = 0;
    }
    stage0LayoutEntries[8].binding = 8;
    stage0LayoutEntries[8].visibility = wgpu::ShaderStage::Compute;
    stage0LayoutEntries[8].buffer.type = wgpu::BufferBindingType::Uniform;
    stage0LayoutEntries[8].buffer.minBindingSize = sizeof(Stage0ParamsData);
    wgpu::BindGroupLayoutDescriptor stage0BglDesc = {};
    stage0BglDesc.entryCount = 9;
    stage0BglDesc.entries = stage0LayoutEntries;
    session.stage0BindGroupLayout = session.device.CreateBindGroupLayout(&stage0BglDesc);

    wgpu::PipelineLayoutDescriptor stage0PlDesc = {};
    stage0PlDesc.bindGroupLayoutCount = 1;
    stage0PlDesc.bindGroupLayouts = &session.stage0BindGroupLayout;
    session.stage0PipelineLayout = session.device.CreatePipelineLayout(&stage0PlDesc);

    wgpu::BindGroupLayoutEntry stage0ScoreLayoutEntries[4] = {};
    for (std::uint32_t i = 0; i < 3; ++i) {
        stage0ScoreLayoutEntries[i].binding = i;
        stage0ScoreLayoutEntries[i].visibility = wgpu::ShaderStage::Compute;
        stage0ScoreLayoutEntries[i].buffer.type =
            (i <= 1u) ? wgpu::BufferBindingType::ReadOnlyStorage
                      : wgpu::BufferBindingType::Storage;
    }
    stage0ScoreLayoutEntries[3].binding = 3;
    stage0ScoreLayoutEntries[3].visibility = wgpu::ShaderStage::Compute;
    stage0ScoreLayoutEntries[3].buffer.type = wgpu::BufferBindingType::Uniform;
    stage0ScoreLayoutEntries[3].buffer.minBindingSize = sizeof(Stage0ParamsData);
    wgpu::BindGroupLayoutDescriptor stage0ScoreBglDesc = {};
    stage0ScoreBglDesc.entryCount = 4;
    stage0ScoreBglDesc.entries = stage0ScoreLayoutEntries;
    session.stage0ScoreBindGroupLayout =
        session.device.CreateBindGroupLayout(&stage0ScoreBglDesc);

    wgpu::PipelineLayoutDescriptor stage0ScorePlDesc = {};
    stage0ScorePlDesc.bindGroupLayoutCount = 1;
    stage0ScorePlDesc.bindGroupLayouts = &session.stage0ScoreBindGroupLayout;
    session.stage0ScorePipelineLayout =
        session.device.CreatePipelineLayout(&stage0ScorePlDesc);

    wgpu::BindGroupLayoutEntry downsampleLayoutEntries[3] = {};
    downsampleLayoutEntries[0].binding = 0;
    downsampleLayoutEntries[0].visibility = wgpu::ShaderStage::Compute;
    downsampleLayoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    downsampleLayoutEntries[1].binding = 1;
    downsampleLayoutEntries[1].visibility = wgpu::ShaderStage::Compute;
    downsampleLayoutEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
    downsampleLayoutEntries[2].binding = 2;
    downsampleLayoutEntries[2].visibility = wgpu::ShaderStage::Compute;
    downsampleLayoutEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
    downsampleLayoutEntries[2].buffer.minBindingSize = 4u * sizeof(std::uint32_t);
    wgpu::BindGroupLayoutDescriptor downsampleBglDesc = {};
    downsampleBglDesc.entryCount = 3;
    downsampleBglDesc.entries = downsampleLayoutEntries;
    session.downsampleBindGroupLayout =
        session.device.CreateBindGroupLayout(&downsampleBglDesc);

    wgpu::PipelineLayoutDescriptor downsamplePlDesc = {};
    downsamplePlDesc.bindGroupLayoutCount = 1;
    downsamplePlDesc.bindGroupLayouts = &session.downsampleBindGroupLayout;
    session.downsamplePipelineLayout =
        session.device.CreatePipelineLayout(&downsamplePlDesc);

    const auto finishCreatePipelineLayouts = std::chrono::steady_clock::now();
    session.initProfiling.createPipelineLayoutsTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(finishCreatePipelineLayouts - startCreatePipelineLayouts);

    const auto startCreatePSO = std::chrono::high_resolution_clock::now();
    wgpu::ComputePipelineDescriptor preprocessPipeDesc = {};
    preprocessPipeDesc.layout = session.preprocessPipelineLayout;
    preprocessPipeDesc.compute.module = session.preprocessShader;
    preprocessPipeDesc.compute.entryPoint = "main";
    session.preprocessPipeline = session.device.CreateComputePipeline(&preprocessPipeDesc);

    if (enableDebugPipeline) {
        wgpu::ComputePipelineDescriptor stage0PipeDesc = {};
        stage0PipeDesc.layout = session.stage0PipelineLayout;
        stage0PipeDesc.compute.module = session.stage0Shader;
        stage0PipeDesc.compute.entryPoint = "main";
        session.stage0Pipeline = session.device.CreateComputePipeline(&stage0PipeDesc);
    } else {
        wgpu::ComputePipelineDescriptor stage0ScorePipeDesc = {};
        stage0ScorePipeDesc.layout = session.stage0ScorePipelineLayout;
        stage0ScorePipeDesc.compute.module = session.stage0ScoreShader;
        stage0ScorePipeDesc.compute.entryPoint = "main";
        session.stage0ScorePipeline =
            session.device.CreateComputePipeline(&stage0ScorePipeDesc);
    }

    wgpu::ComputePipelineDescriptor rgba8ToLinearPipeDesc = {};
    rgba8ToLinearPipeDesc.layout = session.rgba8ToLinearPipelineLayout;
    rgba8ToLinearPipeDesc.compute.module = session.rgba8ToLinearShader;
    rgba8ToLinearPipeDesc.compute.entryPoint = "main";
    session.rgba8ToLinearPipeline =
        session.device.CreateComputePipeline(&rgba8ToLinearPipeDesc);

    wgpu::ComputePipelineDescriptor downsamplePipeDesc = {};
    downsamplePipeDesc.layout = session.downsamplePipelineLayout;
    downsamplePipeDesc.compute.module = session.downsampleShader;
    downsamplePipeDesc.compute.entryPoint = "main";
    session.downsamplePipeline = session.device.CreateComputePipeline(&downsamplePipeDesc);

    const auto finishCreatePSO = std::chrono::high_resolution_clock::now();
    session.initProfiling.createPSOTime = duration_cast<milliseconds>(finishCreatePSO - startCreatePSO);

    if (!session.rgba8ToLinearBindGroupLayout ||
        !session.rgba8ToLinearPipelineLayout || !session.rgba8ToLinearPipeline ||
        !session.preprocessBindGroupLayout || !session.preprocessPipelineLayout || !session.preprocessPipeline ||
        (enableDebugPipeline
             ? (!session.stage0BindGroupLayout || !session.stage0PipelineLayout ||
                !session.stage0Pipeline)
             : (!session.stage0ScoreBindGroupLayout || !session.stage0ScorePipelineLayout ||
                !session.stage0ScorePipeline)) ||
        !session.downsampleBindGroupLayout || !session.downsamplePipelineLayout ||
        !session.downsamplePipeline) {
        throw std::runtime_error("failed to create reusable compute pipelines");
    }

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
    bool collectDebugData) {
    if (width == 0u || height == 0u) {
        throw std::runtime_error("RGBA8 comparison dimensions must be non-zero");
    }
    const std::size_t expectedBytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
    if (rgba1.size() != expectedBytes || rgba2.size() != expectedBytes) {
        throw std::runtime_error(
            "RGBA8 comparison buffer size does not match width and height");
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
    const bool identicalInput = std::equal(rgba1.begin(), rgba1.end(), rgba2.begin());

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
                pyramidHeights);
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
    };
    return result;
}

void RunComparison(
    GpuSession& session,
    const CliOptions& options,
    const ComparisonRequest& request) {
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
        const auto stage0ShaderPath = ResolveShaderPath(argv[0], "stage0_absdiff.wgsl");
        const auto stage0ScoreShaderPath = ResolveShaderPath(argv[0], "stage0_score.wgsl");
        const auto labPreprocessShaderPath = ResolveShaderPath(argv[0], "lab_preprocess.wgsl");
        const auto downsampleShaderPath = ResolveShaderPath(argv[0], "downsample_2x2.wgsl");
        const auto rgba8ToLinearShaderPath = ResolveShaderPath(argv[0], "rgba8_to_linear.wgsl");
        const auto stage0ShaderSource = ReadAllText(stage0ShaderPath);
        const auto stage0ScoreShaderSource = ReadAllText(stage0ScoreShaderPath);
        const auto labPreprocessShaderSource = ReadAllText(labPreprocessShaderPath);
        const auto downsampleShaderSource = ReadAllText(downsampleShaderPath);
        const auto rgba8ToLinearShaderSource = ReadAllText(rgba8ToLinearShaderPath);
        GpuSession session =
            CreateGpuSession(
                labPreprocessShaderSource,
                stage0ShaderSource,
                stage0ScoreShaderSource,
                downsampleShaderSource,
                rgba8ToLinearShaderSource,
                options.debugDumpEnabled,
                options.profilingEnabled || !options.out.empty());
        if (options.profilingEnabled) {
            PrintProfilingBuckets(
                BuildSessionInitProfilingBuckets(session.initProfiling),
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
                    RunComparison(session, options, request);
                } catch (const std::exception& ex) {
                    throw std::runtime_error(
                        "comparison failed at line " + std::to_string(lineNumber) + ": " + ex.what());
                }
            }
        } else {
            RunComparison(
                session,
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

