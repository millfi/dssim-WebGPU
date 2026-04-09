#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <numeric>

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
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;
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
};

struct GpuSession {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    std::string adapterName = "unknown";

    wgpu::ShaderModule preprocessShader;
    wgpu::ShaderModule stage0Shader;
    wgpu::ShaderModule downsampleShader;

    wgpu::BindGroupLayout preprocessBindGroupLayout;
    wgpu::PipelineLayout preprocessPipelineLayout;
    wgpu::ComputePipeline preprocessPipeline;

    wgpu::BindGroupLayout stage0BindGroupLayout;
    wgpu::PipelineLayout stage0PipelineLayout;
    wgpu::ComputePipeline stage0Pipeline;

    wgpu::BindGroupLayout downsampleBindGroupLayout;
    wgpu::PipelineLayout downsamplePipelineLayout;
    wgpu::ComputePipeline downsamplePipeline;

    ProfilingSummary initProfiling;
};

struct ProfilingBuckets {
    std::chrono::milliseconds totalTime{0};
    std::chrono::milliseconds pipelineSetupTime{0};
    std::chrono::milliseconds resourcePrepTime{0};
    std::chrono::milliseconds gpuExecutionTime{0};
    std::chrono::milliseconds cpuPostProcessTime{0};
    std::chrono::milliseconds otherTime{0};
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

std::vector<LinearRgba> ConvertRgba8ToLinearPlu(const std::vector<std::uint8_t>& bytes) {
    if ((bytes.size() % 4) != 0) {
        throw std::runtime_error("rgba8 byte count is not divisible by 4");
    }

    const std::size_t pixelCount = bytes.size() / 4;
    std::vector<LinearRgba> out(pixelCount);
    for (std::size_t i = 0; i < pixelCount; ++i) {
        const std::size_t base = i * 4;
        const float a = static_cast<float>(bytes[base + 3]) / 255.0f;
        out[i].r = SrgbToLinear(static_cast<float>(bytes[base + 0]) / 255.0f) * a;
        out[i].g = SrgbToLinear(static_cast<float>(bytes[base + 1]) / 255.0f) * a;
        out[i].b = SrgbToLinear(static_cast<float>(bytes[base + 2]) / 255.0f) * a;
        out[i].a = a;
    }
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

void WriteU8Buffer(const std::filesystem::path& outPath, const std::vector<std::uint8_t>& values) {
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
        os << "        \"elem_count\": " << scale.ssimMap.size() << ",\n";
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

std::vector<std::uint8_t> ReadBufferBlocking(
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
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (mapState.status != wgpu::MapAsyncStatus::Success) {
        std::string message = "readback MapAsync failed";
        if (!mapState.message.empty()) {
            message += ": ";
            message += mapState.message;
        }
        throw std::runtime_error(message);
    }

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

ScaleOutputs RunStage0Compute(
    const GpuSession& session,
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
    const std::size_t labBytes = elemCount * sizeof(float) * 4u;
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
    const auto start_CreateBuffers = std::chrono::steady_clock::now();

    wgpu::BufferDescriptor rgbaStorageDesc = {};
    rgbaStorageDesc.size = static_cast<std::uint64_t>(rgbaBytes);
    rgbaStorageDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    rgbaStorageDesc.mappedAtCreation = false;

    wgpu::Buffer input1Buffer = session.device.CreateBuffer(&rgbaStorageDesc);
    wgpu::Buffer input2Buffer = session.device.CreateBuffer(&rgbaStorageDesc);
    wgpu::BufferDescriptor labStorageDesc = {};
    labStorageDesc.size = static_cast<std::uint64_t>(labBytes);
    labStorageDesc.usage = wgpu::BufferUsage::Storage;
    labStorageDesc.mappedAtCreation = false;
    wgpu::Buffer lab1Buffer = session.device.CreateBuffer(&labStorageDesc);
    wgpu::Buffer lab2Buffer = session.device.CreateBuffer(&labStorageDesc);

    wgpu::BufferDescriptor f32StorageDesc = {};
    f32StorageDesc.size = static_cast<std::uint64_t>(f32Bytes);
    f32StorageDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    f32StorageDesc.mappedAtCreation = false;

    wgpu::Buffer outSsimBuffer = session.device.CreateBuffer(&f32StorageDesc);
    wgpu::Buffer outMu1Buffer = session.device.CreateBuffer(&f32StorageDesc);
    wgpu::Buffer outMu2Buffer = session.device.CreateBuffer(&f32StorageDesc);
    wgpu::Buffer outVar1Buffer = session.device.CreateBuffer(&f32StorageDesc);
    wgpu::Buffer outVar2Buffer = session.device.CreateBuffer(&f32StorageDesc);
    wgpu::Buffer outCov12Buffer = session.device.CreateBuffer(&f32StorageDesc);
    if (!input1Buffer || !input2Buffer || !lab1Buffer || !lab2Buffer || !outSsimBuffer || !outMu1Buffer ||
        !outMu2Buffer || !outVar1Buffer || !outVar2Buffer || !outCov12Buffer) {
        throw std::runtime_error("failed to create stage0 buffers");
    }

    wgpu::BufferDescriptor readbackF32SsimDesc = {};
    readbackF32SsimDesc.size = static_cast<std::uint64_t>(f32Bytes);
    readbackF32SsimDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    readbackF32SsimDesc.mappedAtCreation = false;
    wgpu::Buffer readbackSsimBuffer = session.device.CreateBuffer(&readbackF32SsimDesc);
    if (!readbackSsimBuffer) {
        throw std::runtime_error("failed to create stage0 ssim readback buffer");
    }

    wgpu::Buffer readbackMu1Buffer;
    wgpu::Buffer readbackMu2Buffer;
    wgpu::Buffer readbackVar1Buffer;
    wgpu::Buffer readbackVar2Buffer;
    wgpu::Buffer readbackCov12Buffer;
    if (readIntermediateStats) {
        wgpu::BufferDescriptor readbackF32Desc = {};
        readbackF32Desc.size = static_cast<std::uint64_t>(f32Bytes);
        readbackF32Desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        readbackF32Desc.mappedAtCreation = false;
        readbackMu1Buffer = session.device.CreateBuffer(&readbackF32Desc);
        readbackMu2Buffer = session.device.CreateBuffer(&readbackF32Desc);
        readbackVar1Buffer = session.device.CreateBuffer(&readbackF32Desc);
        readbackVar2Buffer = session.device.CreateBuffer(&readbackF32Desc);
        readbackCov12Buffer = session.device.CreateBuffer(&readbackF32Desc);
        if (!readbackMu1Buffer || !readbackMu2Buffer || !readbackVar1Buffer || !readbackVar2Buffer ||
            !readbackCov12Buffer) {
            throw std::runtime_error("failed to create stage0 stats readback buffers");
        }
    }

    wgpu::BufferDescriptor paramsDesc = {};
    paramsDesc.size = static_cast<std::uint64_t>(sizeof(ParamsData));
    paramsDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    paramsDesc.mappedAtCreation = false;
    wgpu::Buffer paramsBuffer = session.device.CreateBuffer(&paramsDesc);
    if (!paramsBuffer) {
        throw std::runtime_error("failed to create stage0 params buffer");
    }
    const auto finish_CreateBuffers = std::chrono::steady_clock::now();
    outputs.createBuffers_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_CreateBuffers - start_CreateBuffers);

    wgpu::Queue queue = session.device.GetQueue();
    const auto start_WriteInputBuffers = std::chrono::steady_clock::now();
    queue.WriteBuffer(input1Buffer, 0, input1.data(), rgbaBytes);
    queue.WriteBuffer(input2Buffer, 0, input2.data(), rgbaBytes);
    queue.WriteBuffer(paramsBuffer, 0, &paramsData, sizeof(ParamsData));
    const auto finish_WriteInputBuffers = std::chrono::steady_clock::now();
    outputs.writeInputBuffers_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_WriteInputBuffers - start_WriteInputBuffers);

    if (!session.preprocessShader || !session.stage0Shader) {
        throw std::runtime_error("failed to create stage0/preprocess shader module");
    }
    if (!session.preprocessBindGroupLayout || !session.preprocessPipelineLayout || !session.preprocessPipeline) {
        throw std::runtime_error("failed to create preprocess pipeline");
    }
    const auto start_CreateBindGroups = std::chrono::steady_clock::now();

    wgpu::BindGroupEntry preprocessBg1Entries[3] = {};
    preprocessBg1Entries[0].binding = 0;
    preprocessBg1Entries[0].buffer = input1Buffer;
    preprocessBg1Entries[0].size = static_cast<std::uint64_t>(rgbaBytes);
    preprocessBg1Entries[1].binding = 1;
    preprocessBg1Entries[1].buffer = lab1Buffer;
    preprocessBg1Entries[1].size = static_cast<std::uint64_t>(labBytes);
    preprocessBg1Entries[2].binding = 2;
    preprocessBg1Entries[2].buffer = paramsBuffer;
    preprocessBg1Entries[2].size = static_cast<std::uint64_t>(sizeof(ParamsData));

    wgpu::BindGroupEntry preprocessBg2Entries[3] = {};
    preprocessBg2Entries[0].binding = 0;
    preprocessBg2Entries[0].buffer = input2Buffer;
    preprocessBg2Entries[0].size = static_cast<std::uint64_t>(rgbaBytes);
    preprocessBg2Entries[1].binding = 1;
    preprocessBg2Entries[1].buffer = lab2Buffer;
    preprocessBg2Entries[1].size = static_cast<std::uint64_t>(labBytes);
    preprocessBg2Entries[2].binding = 2;
    preprocessBg2Entries[2].buffer = paramsBuffer;
    preprocessBg2Entries[2].size = static_cast<std::uint64_t>(sizeof(ParamsData));

    wgpu::BindGroupDescriptor preprocessBg1Desc = {};
    preprocessBg1Desc.layout = session.preprocessBindGroupLayout;
    preprocessBg1Desc.entryCount = 3;
    preprocessBg1Desc.entries = preprocessBg1Entries;
    wgpu::BindGroup preprocessBg1 = session.device.CreateBindGroup(&preprocessBg1Desc);
    wgpu::BindGroupDescriptor preprocessBg2Desc = {};
    preprocessBg2Desc.layout = session.preprocessBindGroupLayout;
    preprocessBg2Desc.entryCount = 3;
    preprocessBg2Desc.entries = preprocessBg2Entries;
    wgpu::BindGroup preprocessBg2 = session.device.CreateBindGroup(&preprocessBg2Desc);
    if (!preprocessBg1 || !preprocessBg2) {
        throw std::runtime_error("failed to create preprocess bind groups");
    }

    if (!session.stage0BindGroupLayout || !session.stage0PipelineLayout || !session.stage0Pipeline) {
        throw std::runtime_error("failed to create stage0 compute pipeline");
    }

    wgpu::BindGroupEntry bgEntries[9] = {};
    bgEntries[0].binding = 0;
    bgEntries[0].buffer = lab1Buffer;
    bgEntries[0].offset = 0;
    bgEntries[0].size = static_cast<std::uint64_t>(labBytes);

    bgEntries[1].binding = 1;
    bgEntries[1].buffer = lab2Buffer;
    bgEntries[1].offset = 0;
    bgEntries[1].size = static_cast<std::uint64_t>(labBytes);

    bgEntries[2].binding = 2;
    bgEntries[2].buffer = outSsimBuffer;
    bgEntries[2].offset = 0;
    bgEntries[2].size = static_cast<std::uint64_t>(f32Bytes);

    bgEntries[3].binding = 3;
    bgEntries[3].buffer = outMu1Buffer;
    bgEntries[3].offset = 0;
    bgEntries[3].size = static_cast<std::uint64_t>(f32Bytes);

    bgEntries[4].binding = 4;
    bgEntries[4].buffer = outMu2Buffer;
    bgEntries[4].offset = 0;
    bgEntries[4].size = static_cast<std::uint64_t>(f32Bytes);

    bgEntries[5].binding = 5;
    bgEntries[5].buffer = outVar1Buffer;
    bgEntries[5].offset = 0;
    bgEntries[5].size = static_cast<std::uint64_t>(f32Bytes);

    bgEntries[6].binding = 6;
    bgEntries[6].buffer = outVar2Buffer;
    bgEntries[6].offset = 0;
    bgEntries[6].size = static_cast<std::uint64_t>(f32Bytes);

    bgEntries[7].binding = 7;
    bgEntries[7].buffer = outCov12Buffer;
    bgEntries[7].offset = 0;
    bgEntries[7].size = static_cast<std::uint64_t>(f32Bytes);

    bgEntries[8].binding = 8;
    bgEntries[8].buffer = paramsBuffer;
    bgEntries[8].offset = 0;
    bgEntries[8].size = static_cast<std::uint64_t>(sizeof(ParamsData));

    wgpu::BindGroupDescriptor bgDesc = {};
    bgDesc.layout = session.stage0BindGroupLayout;
    bgDesc.entryCount = 9;
    bgDesc.entries = bgEntries;
    wgpu::BindGroup bindGroup = session.device.CreateBindGroup(&bgDesc);
    if (!bindGroup) {
        throw std::runtime_error("failed to create stage0 bind group");
    }
    const auto finish_CreateBindGroups = std::chrono::steady_clock::now();
    outputs.createBindGroups_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_CreateBindGroups - start_CreateBindGroups);
    const auto start_DispatchAndSubmit = std::chrono::steady_clock::now();

    const std::uint32_t wgX = (width + 15u) / 16u;
    const std::uint32_t wgY = (height + 15u) / 16u;

    wgpu::CommandEncoder encoder = session.device.CreateCommandEncoder();
    {
        wgpu::ComputePassDescriptor passDesc = {};
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&passDesc);
        pass.SetPipeline(session.preprocessPipeline);
        pass.SetBindGroup(0, preprocessBg1);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.SetBindGroup(0, preprocessBg2);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.End();
    }
    {
        wgpu::ComputePassDescriptor passDesc = {};
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&passDesc);
        pass.SetPipeline(session.stage0Pipeline);
        pass.SetBindGroup(0, bindGroup);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.End();
    }
    encoder.CopyBufferToBuffer(outSsimBuffer, 0, readbackSsimBuffer, 0, static_cast<std::uint64_t>(f32Bytes));
    if (readIntermediateStats) {
        encoder.CopyBufferToBuffer(outMu1Buffer, 0, readbackMu1Buffer, 0, static_cast<std::uint64_t>(f32Bytes));
        encoder.CopyBufferToBuffer(outMu2Buffer, 0, readbackMu2Buffer, 0, static_cast<std::uint64_t>(f32Bytes));
        encoder.CopyBufferToBuffer(outVar1Buffer, 0, readbackVar1Buffer, 0, static_cast<std::uint64_t>(f32Bytes));
        encoder.CopyBufferToBuffer(outVar2Buffer, 0, readbackVar2Buffer, 0, static_cast<std::uint64_t>(f32Bytes));
        encoder.CopyBufferToBuffer(outCov12Buffer, 0, readbackCov12Buffer, 0, static_cast<std::uint64_t>(f32Bytes));
    }

    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    queue.Submit(1, &commandBuffer);
    const auto finish_DispatchAndSubmit = std::chrono::steady_clock::now();
    outputs.dispatchAndSubmit_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_DispatchAndSubmit - start_DispatchAndSubmit);

    
    outputs.width = width;
    outputs.height = height;
    const auto start_Readback = std::chrono::steady_clock::now();
    const auto ssimBytes = ReadBufferBlocking(session.instance, readbackSsimBuffer, f32Bytes);
    outputs.ssimMap.resize(elemCount);
    std::memcpy(outputs.ssimMap.data(), ssimBytes.data(), f32Bytes);
    if (readIntermediateStats) {
        const auto mu1Bytes = ReadBufferBlocking(session.instance, readbackMu1Buffer, f32Bytes);
        const auto mu2Bytes = ReadBufferBlocking(session.instance, readbackMu2Buffer, f32Bytes);
        const auto var1Bytes = ReadBufferBlocking(session.instance, readbackVar1Buffer, f32Bytes);
        const auto var2Bytes = ReadBufferBlocking(session.instance, readbackVar2Buffer, f32Bytes);
        const auto cov12Bytes = ReadBufferBlocking(session.instance, readbackCov12Buffer, f32Bytes);
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
    const auto finish_Readback = std::chrono::steady_clock::now();
    outputs.readback_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_Readback - start_Readback);

    const auto start_PostProcess = std::chrono::steady_clock::now();
    // Aggregate f32 SSIM values in f64, matching the reference implementation.
    double ssimSum = 0.0;
    for (float s : outputs.ssimMap) {
        ssimSum += static_cast<double>(s);
    }
    outputs.meanSsim = ssimSum / static_cast<double>(elemCount);
    const double avg =
        std::pow(std::max(outputs.meanSsim, 0.0), std::pow(0.5, static_cast<double>(scaleLevel)));
    double devSum = 0.0;
    for (float s : outputs.ssimMap) {
        devSum += std::abs(avg - static_cast<double>(s));
    }
    outputs.ssimScore = 1.0 - (devSum / static_cast<double>(elemCount));
    const auto finish_PostProcess = std::chrono::steady_clock::now();
    outputs.postProcess_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish_PostProcess - start_PostProcess);
    return outputs;
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

wgpu::Device RequestDeviceBlocking(const wgpu::Instance& instance, const wgpu::Adapter& adapter) {
    struct RequestState {
        std::atomic<bool> done{false};
        wgpu::RequestDeviceStatus status = wgpu::RequestDeviceStatus::Error;
        wgpu::Device device = nullptr;
        std::string message;
    };
    RequestState state;

    adapter.RequestDevice(
        nullptr,
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
    const std::string& downsampleShaderSource) {
    GpuSession session;

    dawnProcSetProcs(&dawn::native::GetProcs());

    session.instance = wgpu::CreateInstance();
    if (!session.instance) {
        throw std::runtime_error("failed to create WGPU instance");
    }

    session.adapter = RequestAdapterBlocking(session.instance);
    session.device = RequestDeviceBlocking(session.instance, session.adapter);

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
    session.stage0Shader = CreateShaderModule(session.device, stage0ShaderSource);
    session.downsampleShader = CreateShaderModule(session.device, downsampleShaderSource);
    const auto finishCreateShaderModule = std::chrono::steady_clock::now();
    session.initProfiling.createShaderModuleTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(finishCreateShaderModule - startCreateShaderModule);
    if (!session.preprocessShader || !session.stage0Shader || !session.downsampleShader) {
        throw std::runtime_error("failed to create reusable shader modules");
    }

    struct Stage0ParamsData {
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

    const auto startCreatePipelineLayouts = std::chrono::steady_clock::now();

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
    downsampleLayoutEntries[2].buffer.minBindingSize = sizeof(DownsampleParamsData);
    wgpu::BindGroupLayoutDescriptor downsampleBglDesc = {};
    downsampleBglDesc.entryCount = 3;
    downsampleBglDesc.entries = downsampleLayoutEntries;
    session.downsampleBindGroupLayout = session.device.CreateBindGroupLayout(&downsampleBglDesc);

    wgpu::PipelineLayoutDescriptor downsamplePlDesc = {};
    downsamplePlDesc.bindGroupLayoutCount = 1;
    downsamplePlDesc.bindGroupLayouts = &session.downsampleBindGroupLayout;
    session.downsamplePipelineLayout = session.device.CreatePipelineLayout(&downsamplePlDesc);

    const auto finishCreatePipelineLayouts = std::chrono::steady_clock::now();
    session.initProfiling.createPipelineLayoutsTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(finishCreatePipelineLayouts - startCreatePipelineLayouts);

    const auto startCreatePSO = std::chrono::high_resolution_clock::now();
    wgpu::ComputePipelineDescriptor preprocessPipeDesc = {};
    preprocessPipeDesc.layout = session.preprocessPipelineLayout;
    preprocessPipeDesc.compute.module = session.preprocessShader;
    preprocessPipeDesc.compute.entryPoint = "main";
    session.preprocessPipeline = session.device.CreateComputePipeline(&preprocessPipeDesc);

    wgpu::ComputePipelineDescriptor stage0PipeDesc = {};
    stage0PipeDesc.layout = session.stage0PipelineLayout;
    stage0PipeDesc.compute.module = session.stage0Shader;
    stage0PipeDesc.compute.entryPoint = "main";
    session.stage0Pipeline = session.device.CreateComputePipeline(&stage0PipeDesc);

    wgpu::ComputePipelineDescriptor downsamplePipeDesc = {};
    downsamplePipeDesc.layout = session.downsamplePipelineLayout;
    downsamplePipeDesc.compute.module = session.downsampleShader;
    downsamplePipeDesc.compute.entryPoint = "main";
    session.downsamplePipeline = session.device.CreateComputePipeline(&downsamplePipeDesc);
    const auto finishCreatePSO = std::chrono::high_resolution_clock::now();
    session.initProfiling.createPSOTime = duration_cast<milliseconds>(finishCreatePSO - startCreatePSO);

    if (!session.preprocessBindGroupLayout || !session.preprocessPipelineLayout || !session.preprocessPipeline ||
        !session.stage0BindGroupLayout || !session.stage0PipelineLayout || !session.stage0Pipeline ||
        !session.downsampleBindGroupLayout || !session.downsamplePipelineLayout || !session.downsamplePipeline) {
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
        .totalTime = std::chrono::milliseconds(profiling.decodeDoneToScoreMs),
        .pipelineSetupTime =
            profiling.createShaderModuleTime + profiling.createPipelineLayoutsTime + profiling.createPSOTime,
        .resourcePrepTime =
            profiling.createBuffersTime + profiling.writeInputBuffersTime + profiling.createBindGroupsTime,
        .gpuExecutionTime = profiling.dispatchAndSubmitTime + profiling.readbackTime,
        .cpuPostProcessTime = profiling.postProcessTime,
        .otherTime = profiling.otherTime,
    };
}

ProfilingBuckets BuildSessionInitProfilingBuckets(const ProfilingSummary& profiling) {
    const auto totalTime =
        profiling.createShaderModuleTime + profiling.createPipelineLayoutsTime + profiling.createPSOTime;
    return {
        .totalTime = totalTime,
        .pipelineSetupTime = totalTime,
        .resourcePrepTime = std::chrono::milliseconds{0},
        .gpuExecutionTime = std::chrono::milliseconds{0},
        .cpuPostProcessTime = std::chrono::milliseconds{0},
        .otherTime = profiling.otherTime,
    };
}

void PrintProfilingBuckets(
    const ProfilingBuckets& buckets,
    const std::string_view prefix,
    const std::string_view label) {
    std::cout << prefix << label << "total_ms = " << buckets.totalTime.count() << '\n';
    std::cout << prefix << label << "pipeline_setup_ms = " << buckets.pipelineSetupTime.count() << '\n';
    std::cout << prefix << label << "resource_prep_ms = " << buckets.resourcePrepTime.count() << '\n';
    std::cout << prefix << label << "gpu_execution_ms = " << buckets.gpuExecutionTime.count() << '\n';
    std::cout << prefix << label << "cpu_postprocess_ms = " << buckets.cpuPostProcessTime.count() << '\n';
    std::cout << prefix << label << "other_ms = " << buckets.otherTime.count() << '\n';
}

void RunComparison(
    const GpuSession& session,
    const CliOptions& options,
    const ComparisonRequest& request) {
    const DecodedImage image1 = LoadPngRgba8(request.image1);
    const DecodedImage image2 = LoadPngRgba8(request.image2);
    if (image1.pixels.empty() || image2.pixels.empty()) {
        throw std::runtime_error("decoded png pixels are empty");
    }
    if (image1.width != image2.width || image1.height != image2.height) {
        throw std::runtime_error("image size mismatch; multi-scale stage requires identical dimensions");
    }
    const auto decodeDoneAt = std::chrono::steady_clock::now();

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

    const auto input1 = ConvertRgba8ToLinearPlu(image1.pixels);
    const auto input2 = ConvertRgba8ToLinearPlu(image2.pixels);

    // Identical input produces score 0 by definition (matches reference behavior).
    // GPU dispatches may introduce f32 non-determinism between separate runs,
    // so we detect this case on the CPU side.
    const bool identicalInput = (image1.pixels == image2.pixels);

    MultiScaleOutputs compute;
    std::vector<LinearRgba> curr1 = input1;
    std::vector<LinearRgba> curr2 = input2;
    std::uint32_t currWidth = image1.width;
    std::uint32_t currHeight = image1.height;

    DownsampleOutputs firstDownsample1;
    DownsampleOutputs firstDownsample2;

    milliseconds createShaderModuleProcessingTime{0};
    milliseconds createPSOProcessingTime{0};
    milliseconds createBuffersProcessingTime{0};
    milliseconds writeInputBuffersProcessingTime{0};
    milliseconds createPipelineLayoutsProcessingTime{0};
    milliseconds createBindGroupsProcessingTime{0};
    milliseconds dispatchAndSubmitProcessingTime{0};
    milliseconds readbackProcessingTime{0};
    milliseconds postProcessProcessingTime{0};
    for (std::size_t level = 0; level < kDefaultScaleWeights.size(); ++level) {
        const bool readStats = options.debugDumpEnabled && level == 0;
        ScaleOutputs scale =
            RunStage0Compute(session, curr1, curr2, currWidth, currHeight, level, readStats);
        createShaderModuleProcessingTime += scale.createShaderModule_time;
        createPSOProcessingTime += scale.createPSO_time;
        createBuffersProcessingTime += scale.createBuffers_time;
        writeInputBuffersProcessingTime += scale.writeInputBuffers_time;
        createPipelineLayoutsProcessingTime += scale.createPipelineLayouts_time;
        createBindGroupsProcessingTime += scale.createBindGroups_time;
        dispatchAndSubmitProcessingTime += scale.dispatchAndSubmit_time;
        readbackProcessingTime += scale.readback_time;
        postProcessProcessingTime += scale.postProcess_time;
        compute.scales.push_back(std::move(scale));
        if (level + 1 >= kDefaultScaleWeights.size()) {
            break;
        }
        if (currWidth < 8 || currHeight < 8) {
            break;
        }

        DownsampleOutputs next1 = RunDownsample2x2Compute(session, curr1, currWidth, currHeight);
        DownsampleOutputs next2 = RunDownsample2x2Compute(session, curr2, currWidth, currHeight);
        createShaderModuleProcessingTime += next1.createShaderModule_time + next2.createShaderModule_time;
        createPSOProcessingTime += next1.createPSO_time + next2.createPSO_time;
        createBuffersProcessingTime += next1.createBuffers_time + next2.createBuffers_time;
        writeInputBuffersProcessingTime += next1.writeInputBuffers_time + next2.writeInputBuffers_time;
        createPipelineLayoutsProcessingTime += next1.createPipelineLayouts_time + next2.createPipelineLayouts_time;
        createBindGroupsProcessingTime += next1.createBindGroups_time + next2.createBindGroups_time;
        dispatchAndSubmitProcessingTime += next1.dispatchAndSubmit_time + next2.dispatchAndSubmit_time;
        readbackProcessingTime += next1.readback_time + next2.readback_time;
        if (level == 0) {
            firstDownsample1 = next1;
            firstDownsample2 = next2;
        }
        currWidth = next1.width;
        currHeight = next1.height;
        curr1 = std::move(next1.pixels);
        curr2 = std::move(next2.pixels);
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
        debugInfo.stage0ElemCount = compute.scales.empty() ? 0 : compute.scales[0].ssimMap.size();
        WriteU8Buffer(debugInfo.image1RgbaPath, image1.pixels);
        WriteU8Buffer(debugInfo.image2RgbaPath, image2.pixels);
        WriteF32LeBuffer(debugInfo.stage0DssimPath, compute.scales[0].ssimMap);
        WriteF32LeBuffer(debugInfo.stage0Mu1Path, compute.scales[0].mu1);
        WriteF32LeBuffer(debugInfo.stage0Mu2Path, compute.scales[0].mu2);
        WriteF32LeBuffer(debugInfo.stage0Var1Path, compute.scales[0].var1);
        WriteF32LeBuffer(debugInfo.stage0Var2Path, compute.scales[0].var2);
        WriteF32LeBuffer(debugInfo.stage0Cov12Path, compute.scales[0].cov12);
        if (compute.scales.size() > 1 && !firstDownsample1.pixels.empty() && !firstDownsample2.pixels.empty()) {
            debugInfo.image1Scale1Path = options.debugDumpDir / "image1_scale1_rgba8.gpu.bin";
            debugInfo.image2Scale1Path = options.debugDumpDir / "image2_scale1_rgba8.gpu.bin";
            debugInfo.stage1DssimPath = options.debugDumpDir / "stage1_dssim5x5_gaussian_linear_u32le.gpu.bin";
            debugInfo.stage1ElemCount = compute.scales[1].ssimMap.size();
            WriteU8Buffer(debugInfo.image1Scale1Path, ConvertLinearPluToRgba8(firstDownsample1.pixels));
            WriteU8Buffer(debugInfo.image2Scale1Path, ConvertLinearPluToRgba8(firstDownsample2.pixels));
            WriteF32LeBuffer(debugInfo.stage1DssimPath, compute.scales[1].ssimMap);
        }
        debugInfoPtr = &debugInfo;
    }

    std::ostringstream scoreText;
    scoreText << std::fixed << std::setprecision(8) << compute.score;
    const auto scoreReadyAt = std::chrono::steady_clock::now();
    const auto decodeDoneToScoreMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(scoreReadyAt - decodeDoneAt).count();
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
    CliOptions resultOptions = options;
    resultOptions.image1 = request.image1;
    resultOptions.image2 = request.image2;
    const ProfilingSummary profiling = {
        .decodeDoneToScoreMs = decodeDoneToScoreMs,
        .createShaderModuleTime = createShaderModuleProcessingTime,
        .createPSOTime = createPSOProcessingTime,
        .createBuffersTime = createBuffersProcessingTime,
        .writeInputBuffersTime = writeInputBuffersProcessingTime,
        .createPipelineLayoutsTime = createPipelineLayoutsProcessingTime,
        .createBindGroupsTime = createBindGroupsProcessingTime,
        .dispatchAndSubmitTime = dispatchAndSubmitProcessingTime,
        .readbackTime = readbackProcessingTime,
        .postProcessTime = postProcessProcessingTime,
        .otherTime = milliseconds(decodeDoneToScoreMs) - measuredProcessingTime,
    };

    if (!resultOptions.out.empty()) {
        const std::string json =
            BuildJson(resultOptions, session.adapterName, decoded1, decoded2, compute, profiling, debugInfoPtr);
        WriteStringFile(resultOptions.out, json);
    }

    std::cout << scoreText.str() << '\t' << resultOptions.image2.string() << '\n';
    if (resultOptions.profilingEnabled) {
        PrintProfilingBuckets(BuildRuntimeProfilingBuckets(profiling), "[profiling] ", "");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = ParseArgs(argc, argv);
        const auto stage0ShaderPath = ResolveShaderPath(argv[0], "stage0_absdiff.wgsl");
        const auto downsampleShaderPath = ResolveShaderPath(argv[0], "downsample_2x2.wgsl");
        const auto labPreprocessShaderPath = ResolveShaderPath(argv[0], "lab_preprocess.wgsl");
        const auto stage0ShaderSource = ReadAllText(stage0ShaderPath);
        const auto downsampleShaderSource = ReadAllText(downsampleShaderPath);
        const auto labPreprocessShaderSource = ReadAllText(labPreprocessShaderPath);
        const GpuSession session =
            CreateGpuSession(labPreprocessShaderSource, stage0ShaderSource, downsampleShaderSource);
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

