#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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
#include <vector>

#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp.h>

#include "png_loader.h"

namespace {

constexpr std::uint32_t kStage0QScale = 1000000u;
constexpr std::uint32_t kStage0WindowRadius = 1u;
constexpr std::uint32_t kStage0WindowSize = kStage0WindowRadius * 2u + 1u;

struct CliOptions {
    std::filesystem::path image1;
    std::filesystem::path image2;
    std::filesystem::path out;
    std::filesystem::path debugDumpDir;
    bool debugDumpEnabled = false;
};

struct ComputeOutputs {
    std::vector<std::uint32_t> stage0DssimQ;
    std::uint64_t stage0DssimQSum = 0;
    double score = 0.0;
};

struct DebugDumpInfo {
    std::filesystem::path stage0DssimPath;
    std::filesystem::path image1RgbaPath;
    std::filesystem::path image2RgbaPath;
    std::size_t elemCount = 0;
};

struct DecodedInputInfo {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    std::size_t byteCount = 0;
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
    if (argc < 3) {
        throw std::runtime_error(
            "usage: dssim_gpu_dawn_checksum <img1> <img2> [--out <json>] "
            "[--debug-dump-dir <dir>]");
    }

    CliOptions options;
    options.image1 = argv[1];
    options.image2 = argv[2];

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];

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

        throw std::runtime_error("unknown argument: " + arg);
    }

    if (options.debugDumpEnabled && options.debugDumpDir.empty()) {
        throw std::runtime_error("empty --debug-dump-dir");
    }

    return options;
}

std::vector<std::uint32_t> PackRgba8ToU32(const std::vector<std::uint8_t>& bytes) {
    if ((bytes.size() % 4) != 0) {
        throw std::runtime_error("rgba8 byte count is not divisible by 4");
    }

    const std::size_t pixelCount = bytes.size() / 4;
    std::vector<std::uint32_t> out(pixelCount, 0u);
    for (std::size_t i = 0; i < pixelCount; ++i) {
        const std::size_t base = i * 4;
        out[i] =
            static_cast<std::uint32_t>(bytes[base + 0]) |
            (static_cast<std::uint32_t>(bytes[base + 1]) << 8) |
            (static_cast<std::uint32_t>(bytes[base + 2]) << 16) |
            (static_cast<std::uint32_t>(bytes[base + 3]) << 24);
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
    const ComputeOutputs& compute,
    const DebugDumpInfo* debugInfo) {
    const auto abs1 = std::filesystem::absolute(options.image1).string();
    const auto abs2 = std::filesystem::absolute(options.image2).string();
    const auto absOut = std::filesystem::absolute(options.out).string();

    std::ostringstream command;
    command << "dssim_gpu_dawn_checksum \"" << abs1 << "\" \"" << abs2 << "\" --out \"" << absOut << "\"";
    if (options.debugDumpEnabled) {
        const auto absDebug = std::filesystem::absolute(options.debugDumpDir).string();
        command << " --debug-dump-dir \"" << absDebug << "\"";
    }

    std::ostringstream os;
    os << "{\n";
    os << "  \"schema_version\": 1,\n";
    os << "  \"engine\": \"gpu-dawn-wgsl-dssim-stage3x3\",\n";
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
    os << "  \"version\": \"dawn-dssim-stage3x3-1\",\n";
    os << "  \"result\": {\n";
    std::ostringstream scoreText;
    scoreText << std::fixed << std::setprecision(8) << compute.score;
    os << "    \"score_source\": \"gpu-stage3x3-ssim-provisional\",\n";
    os << "    \"score_text\": \"" << scoreText.str() << "\",\n";
    os << "    \"score_f64\": " << std::setprecision(17) << compute.score << ",\n";
    os << "    \"score_bits_u64\": \"" << ToHexU64(compute.score) << "\",\n";
    os << "    \"compared_path\": \"" << EscapeJson(abs2) << "\",\n";
    os << "    \"gpu_stage0\": {\n";
    os << "      \"metric\": \"dssim_3x3_luma\",\n";
    os << "      \"window_radius\": " << kStage0WindowRadius << ",\n";
    os << "      \"window_size\": " << kStage0WindowSize << ",\n";
    os << "      \"qscale\": " << kStage0QScale << ",\n";
    os << "      \"sum_u64\": " << compute.stage0DssimQSum << ",\n";
    os << "      \"elem_count\": " << compute.stage0DssimQ.size() << ",\n";
    os << "      \"mean_f64\": " << std::setprecision(17) << compute.score << "\n";
    os << "    }\n";
    os << "  },\n";
    os << "  \"adapter\": \"" << EscapeJson(adapterName) << "\"";

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
        os << "    \"stage0_dssim3x3_u32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage0DssimPath).string()) << "\",\n";
        os << "      \"elem_type\": \"u32_le\",\n";
        os << "      \"elem_count\": " << debugInfo->elemCount << "\n";
        os << "    },\n";
        os << "    \"stage0_absdiff_u32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->stage0DssimPath).string()) << "\",\n";
        os << "      \"elem_type\": \"u32_le\",\n";
        os << "      \"elem_count\": " << debugInfo->elemCount << "\n";
        os << "    }\n";
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

ComputeOutputs RunStage0Compute(
    const wgpu::Instance& instance,
    const wgpu::Device& device,
    const std::vector<std::uint32_t>& input1,
    const std::vector<std::uint32_t>& input2,
    std::uint32_t width,
    std::uint32_t height,
    const std::string& shaderSource) {
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

    const std::size_t dataBytes = elemCount * sizeof(std::uint32_t);
    struct ParamsData {
        std::uint32_t len;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t qscale;
    };
    const ParamsData paramsData = {
        static_cast<std::uint32_t>(elemCount),
        width,
        height,
        kStage0QScale,
    };
    const std::size_t paramsBytes = sizeof(ParamsData);

    wgpu::BufferDescriptor storageDesc = {};
    storageDesc.size = static_cast<std::uint64_t>(dataBytes);
    storageDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    storageDesc.mappedAtCreation = false;

    wgpu::Buffer input1Buffer = device.CreateBuffer(&storageDesc);
    wgpu::Buffer input2Buffer = device.CreateBuffer(&storageDesc);
    wgpu::Buffer outputBuffer = device.CreateBuffer(&storageDesc);
    if (!input1Buffer || !input2Buffer || !outputBuffer) {
        throw std::runtime_error("failed to create storage buffers");
    }

    wgpu::BufferDescriptor readbackDesc = {};
    readbackDesc.size = static_cast<std::uint64_t>(dataBytes);
    readbackDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    readbackDesc.mappedAtCreation = false;
    wgpu::Buffer readbackBuffer = device.CreateBuffer(&readbackDesc);
    if (!readbackBuffer) {
        throw std::runtime_error("failed to create readback buffer");
    }

    wgpu::BufferDescriptor paramsDesc = {};
    paramsDesc.size = static_cast<std::uint64_t>(paramsBytes);
    paramsDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    paramsDesc.mappedAtCreation = false;
    wgpu::Buffer paramsBuffer = device.CreateBuffer(&paramsDesc);
    if (!paramsBuffer) {
        throw std::runtime_error("failed to create params buffer");
    }

    wgpu::Queue queue = device.GetQueue();
    queue.WriteBuffer(input1Buffer, 0, input1.data(), dataBytes);
    queue.WriteBuffer(input2Buffer, 0, input2.data(), dataBytes);
    queue.WriteBuffer(paramsBuffer, 0, &paramsData, paramsBytes);

    wgpu::ShaderModule shader = CreateShaderModule(device, shaderSource);
    if (!shader) {
        throw std::runtime_error("failed to create shader module");
    }

    wgpu::BindGroupLayoutEntry layoutEntries[4] = {};
    for (std::uint32_t i = 0; i < 3; ++i) {
        layoutEntries[i].binding = i;
        layoutEntries[i].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[i].buffer.type = (i == 2) ? wgpu::BufferBindingType::Storage
                                                : wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[i].buffer.minBindingSize = 0;
    }
    layoutEntries[3].binding = 3;
    layoutEntries[3].visibility = wgpu::ShaderStage::Compute;
    layoutEntries[3].buffer.type = wgpu::BufferBindingType::Uniform;
    layoutEntries[3].buffer.minBindingSize = sizeof(ParamsData);

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 4;
    bglDesc.entries = layoutEntries;
    wgpu::BindGroupLayout bindGroupLayout = device.CreateBindGroupLayout(&bglDesc);
    if (!bindGroupLayout) {
        throw std::runtime_error("failed to create bind group layout");
    }

    wgpu::PipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &bindGroupLayout;
    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&plDesc);
    if (!pipelineLayout) {
        throw std::runtime_error("failed to create pipeline layout");
    }

    wgpu::ComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.compute.module = shader;
    pipelineDesc.compute.entryPoint = "main";
    wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&pipelineDesc);
    if (!pipeline) {
        throw std::runtime_error("failed to create compute pipeline");
    }

    wgpu::BindGroupEntry bgEntries[4] = {};
    bgEntries[0].binding = 0;
    bgEntries[0].buffer = input1Buffer;
    bgEntries[0].offset = 0;
    bgEntries[0].size = static_cast<std::uint64_t>(dataBytes);

    bgEntries[1].binding = 1;
    bgEntries[1].buffer = input2Buffer;
    bgEntries[1].offset = 0;
    bgEntries[1].size = static_cast<std::uint64_t>(dataBytes);

    bgEntries[2].binding = 2;
    bgEntries[2].buffer = outputBuffer;
    bgEntries[2].offset = 0;
    bgEntries[2].size = static_cast<std::uint64_t>(dataBytes);

    bgEntries[3].binding = 3;
    bgEntries[3].buffer = paramsBuffer;
    bgEntries[3].offset = 0;
    bgEntries[3].size = static_cast<std::uint64_t>(paramsBytes);

    wgpu::BindGroupDescriptor bgDesc = {};
    bgDesc.layout = bindGroupLayout;
    bgDesc.entryCount = 4;
    bgDesc.entries = bgEntries;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bgDesc);
    if (!bindGroup) {
        throw std::runtime_error("failed to create bind group");
    }

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    {
        wgpu::ComputePassDescriptor passDesc = {};
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&passDesc);
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bindGroup);
        const std::uint32_t workgroupCount = static_cast<std::uint32_t>((elemCount + 63) / 64);
        pass.DispatchWorkgroups(workgroupCount, 1, 1);
        pass.End();
    }
    encoder.CopyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, static_cast<std::uint64_t>(dataBytes));

    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    queue.Submit(1, &commandBuffer);

    struct MapState {
        std::atomic<bool> done{false};
        wgpu::MapAsyncStatus status = wgpu::MapAsyncStatus::Error;
        std::string message;
    };
    MapState mapState;

    readbackBuffer.MapAsync(
        wgpu::MapMode::Read,
        0,
        static_cast<std::uint64_t>(dataBytes),
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

    const void* mapped = readbackBuffer.GetConstMappedRange(0, static_cast<std::uint64_t>(dataBytes));
    if (mapped == nullptr) {
        throw std::runtime_error("GetConstMappedRange returned null");
    }

    ComputeOutputs outputs;
    outputs.stage0DssimQ.resize(elemCount);
    std::memcpy(outputs.stage0DssimQ.data(), mapped, dataBytes);
    readbackBuffer.Unmap();

    std::uint64_t sum = 0;
    for (std::uint32_t v : outputs.stage0DssimQ) {
        sum += static_cast<std::uint64_t>(v);
    }
    outputs.stage0DssimQSum = sum;
    outputs.score = static_cast<double>(sum) /
                    (static_cast<double>(elemCount) * static_cast<double>(paramsData.qscale));
    return outputs;
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = ParseArgs(argc, argv);
        const auto shaderPath = ResolveShaderPath(argv[0], "stage0_absdiff.wgsl");
        const auto shaderSource = ReadAllText(shaderPath);
        const DecodedImage image1 = LoadPngRgba8(options.image1);
        const DecodedImage image2 = LoadPngRgba8(options.image2);
        if (image1.pixels.empty() || image2.pixels.empty()) {
            throw std::runtime_error("decoded png pixels are empty");
        }
        if (image1.width != image2.width || image1.height != image2.height) {
            throw std::runtime_error("image size mismatch; stage3x3 requires identical dimensions");
        }

        const DecodedInputInfo decoded1 = {
            image1.width,
            image1.height,
            image1.channels,
            image1.pixels.size(),
        };
        const DecodedInputInfo decoded2 = {
            image2.width,
            image2.height,
            image2.channels,
            image2.pixels.size(),
        };

        const auto input1 = PackRgba8ToU32(image1.pixels);
        const auto input2 = PackRgba8ToU32(image2.pixels);

        dawnProcSetProcs(&dawn::native::GetProcs());

        wgpu::Instance instance = wgpu::CreateInstance();
        if (!instance) {
            throw std::runtime_error("failed to create WGPU instance");
        }

        wgpu::Adapter adapter = RequestAdapterBlocking(instance);
        wgpu::Device device = RequestDeviceBlocking(instance, adapter);

        std::string adapterName = "unknown";
        wgpu::AdapterInfo adapterInfo;
        if (adapter.GetInfo(&adapterInfo)) {
            const std::string_view description = static_cast<std::string_view>(adapterInfo.description);
            const std::string_view deviceName = static_cast<std::string_view>(adapterInfo.device);
            if (!description.empty()) {
                adapterName = std::string(description);
            } else if (!deviceName.empty()) {
                adapterName = std::string(deviceName);
            }
        }

        const ComputeOutputs compute = RunStage0Compute(
            instance,
            device,
            input1,
            input2,
            image1.width,
            image1.height,
            shaderSource);

        DebugDumpInfo debugInfo;
        DebugDumpInfo* debugInfoPtr = nullptr;
        if (options.debugDumpEnabled) {
            std::filesystem::create_directories(options.debugDumpDir);
            debugInfo.image1RgbaPath = options.debugDumpDir / "image1_rgba8.gpu.bin";
            debugInfo.image2RgbaPath = options.debugDumpDir / "image2_rgba8.gpu.bin";
            debugInfo.stage0DssimPath = options.debugDumpDir / "stage0_dssim3x3_u32le.gpu.bin";
            debugInfo.elemCount = compute.stage0DssimQ.size();
            WriteU8Buffer(debugInfo.image1RgbaPath, image1.pixels);
            WriteU8Buffer(debugInfo.image2RgbaPath, image2.pixels);
            WriteU32LeBuffer(debugInfo.stage0DssimPath, compute.stage0DssimQ);
            debugInfoPtr = &debugInfo;
        }

        if (!options.out.empty()) {
            const std::string json = BuildJson(options, adapterName, decoded1, decoded2, compute, debugInfoPtr);
            WriteStringFile(options.out, json);
        }

        std::ostringstream scoreText;
        scoreText << std::fixed << std::setprecision(8) << compute.score;
        std::cout << scoreText.str() << '\t' << options.image2.string() << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "dssim_gpu_dawn_checksum error: " << ex.what() << '\n';
        return 1;
    }
}
