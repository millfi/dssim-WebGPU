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

namespace {

struct CliOptions {
    std::filesystem::path image1;
    std::filesystem::path image2;
    std::filesystem::path out;
    std::filesystem::path debugDumpDir;
    bool debugDumpEnabled = false;
};

struct ComputeOutputs {
    std::vector<std::uint32_t> absDiff;
    std::uint64_t absDiffSum = 0;
};

struct DebugDumpInfo {
    std::filesystem::path absDiffPath;
    std::size_t elemCount = 0;
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

std::vector<std::uint8_t> ReadAllBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input: " + path.string());
    }

    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);

    if (size < 0) {
        throw std::runtime_error("failed to get file size: " + path.string());
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input) {
            throw std::runtime_error("failed to read bytes: " + path.string());
        }
    }
    return bytes;
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
    if (argc < 5) {
        throw std::runtime_error(
            "usage: dssim_gpu_dawn_checksum <img1> <img2> --out <json> "
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

    if (options.out.empty()) {
        throw std::runtime_error("missing --out <json>");
    }

    if (options.debugDumpEnabled && options.debugDumpDir.empty()) {
        throw std::runtime_error("empty --debug-dump-dir");
    }

    return options;
}

std::vector<std::uint32_t> PadBytesToU32(const std::vector<std::uint8_t>& bytes, std::size_t len) {
    std::vector<std::uint32_t> out(len, 0u);
    const std::size_t copyLen = std::min(len, bytes.size());
    for (std::size_t i = 0; i < copyLen; ++i) {
        out[i] = static_cast<std::uint32_t>(bytes[i]);
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

std::string BuildJson(
    const CliOptions& options,
    const std::string& adapterName,
    double score,
    const std::string& scoreText,
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
    os << "  \"engine\": \"gpu-dawn-wgsl-byte-absdiff\",\n";
    os << "  \"status\": \"ok\",\n";
    os << "  \"input\": {\n";
    os << "    \"image1\": \"" << EscapeJson(abs1) << "\",\n";
    os << "    \"image2\": \"" << EscapeJson(abs2) << "\"\n";
    os << "  },\n";
    os << "  \"command\": \"" << EscapeJson(command.str()) << "\",\n";
    os << "  \"version\": \"dawn-stage0-absdiff-1\",\n";
    os << "  \"result\": {\n";
    os << "    \"score_text\": \"" << scoreText << "\",\n";
    os << "    \"score_f64\": " << std::setprecision(17) << score << ",\n";
    os << "    \"score_bits_u64\": \"" << ToHexU64(score) << "\",\n";
    os << "    \"compared_path\": \"" << EscapeJson(abs2) << "\"\n";
    os << "  },\n";
    os << "  \"adapter\": \"" << EscapeJson(adapterName) << "\"";

    if (debugInfo != nullptr) {
        os << ",\n";
        os << "  \"debug_dumps\": {\n";
        os << "    \"stage0_absdiff_u32le\": {\n";
        os << "      \"path\": \"" << EscapeJson(std::filesystem::absolute(debugInfo->absDiffPath).string()) << "\",\n";
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

ComputeOutputs RunAbsDiffCompute(
    const wgpu::Instance& instance,
    const wgpu::Device& device,
    const std::vector<std::uint32_t>& input1,
    const std::vector<std::uint32_t>& input2,
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

    const std::size_t dataBytes = elemCount * sizeof(std::uint32_t);
    const std::size_t paramsBytes = sizeof(std::uint32_t);

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
    const std::uint32_t dispatchLen = static_cast<std::uint32_t>(elemCount);
    queue.WriteBuffer(paramsBuffer, 0, &dispatchLen, paramsBytes);

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
    layoutEntries[3].buffer.minBindingSize = sizeof(std::uint32_t);

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
    outputs.absDiff.resize(elemCount);
    std::memcpy(outputs.absDiff.data(), mapped, dataBytes);
    readbackBuffer.Unmap();

    std::uint64_t sum = 0;
    for (std::uint32_t v : outputs.absDiff) {
        sum += static_cast<std::uint64_t>(v);
    }
    outputs.absDiffSum = sum;
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
        const auto bytes1 = ReadAllBytes(options.image1);
        const auto bytes2 = ReadAllBytes(options.image2);
        if (bytes1.empty() || bytes2.empty()) {
            throw std::runtime_error("input image bytes are empty");
        }

        const std::size_t elemCount = std::max(bytes1.size(), bytes2.size());
        const auto input1 = PadBytesToU32(bytes1, elemCount);
        const auto input2 = PadBytesToU32(bytes2, elemCount);

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

        const ComputeOutputs compute = RunAbsDiffCompute(instance, device, input1, input2, shaderSource);
        const double denominator = std::max(1.0, static_cast<double>(elemCount) * 255.0);
        const double score = static_cast<double>(compute.absDiffSum) / denominator;

        DebugDumpInfo debugInfo;
        DebugDumpInfo* debugInfoPtr = nullptr;
        if (options.debugDumpEnabled) {
            std::filesystem::create_directories(options.debugDumpDir);
            debugInfo.absDiffPath = options.debugDumpDir / "stage0_absdiff_u32le.gpu.bin";
            debugInfo.elemCount = compute.absDiff.size();
            WriteU32LeBuffer(debugInfo.absDiffPath, compute.absDiff);
            debugInfoPtr = &debugInfo;
        }

        std::ostringstream scoreText;
        scoreText << std::fixed << std::setprecision(8) << score;

        const std::string json = BuildJson(options, adapterName, score, scoreText.str(), debugInfoPtr);
        WriteStringFile(options.out, json);

        std::cout << scoreText.str() << '\t' << options.image2.string() << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "dssim_gpu_dawn_checksum error: " << ex.what() << '\n';
        return 1;
    }
}
