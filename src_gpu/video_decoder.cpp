#include "video_decoder.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <stdexcept>

namespace {

enum AVPixelFormat SelectVulkanFormat(
    AVCodecContext*,
    const enum AVPixelFormat* formats) {
    for (const enum AVPixelFormat* current = formats;
         current != nullptr && *current != AV_PIX_FMT_NONE;
         ++current) {
        if (*current == AV_PIX_FMT_VULKAN) {
            return *current;
        }
    }
    return AV_PIX_FMT_NONE;
}

void CheckAv(const char* operation, int errorCode) {
    if (errorCode < 0) {
        throw std::runtime_error(std::string(operation) + " failed: " + AvErrorString(errorCode));
    }
}

}  // namespace

std::string AvErrorString(int errorCode) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> buffer{};
    av_strerror(errorCode, buffer.data(), buffer.size());
    return std::string(buffer.data());
}

bool IsVideoPath(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::string extension = path.substr(dot + 1u);
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == "mp4" || extension == "m4v" || extension == "mov" ||
           extension == "mkv" || extension == "webm";
}

std::shared_ptr<VulkanVideoDevice> VulkanVideoDevice::Create(
    const VulkanInteropContext& context) {
    if (context.instance == VK_NULL_HANDLE || context.physicalDevice == VK_NULL_HANDLE ||
        context.device == VK_NULL_HANDLE || context.decodeQueueFamily == VK_QUEUE_FAMILY_IGNORED) {
        throw std::runtime_error("Vulkan Video requires a Vulkan decode queue");
    }

    auto result = std::shared_ptr<VulkanVideoDevice>(new VulkanVideoDevice());
    result->ref_ = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
    if (result->ref_ == nullptr) {
        throw std::runtime_error("av_hwdevice_ctx_alloc(vulkan) failed");
    }

    auto* deviceContext = reinterpret_cast<AVHWDeviceContext*>(result->ref_->data);
    result->vulkan_ = reinterpret_cast<AVVulkanDeviceContext*>(deviceContext->hwctx);
    result->vulkan_->inst = context.instance;
    result->vulkan_->phys_dev = context.physicalDevice;
    result->vulkan_->act_dev = context.device;
    result->vulkan_->get_proc_addr = vkGetInstanceProcAddr;
    result->vulkan_->qf[0].idx = static_cast<int>(context.computeQueueFamily);
    result->vulkan_->qf[0].num = 1;
    result->vulkan_->qf[0].flags =
        static_cast<VkQueueFlagBits>(context.computeQueueFlags);
    result->vulkan_->qf[0].video_caps =
        static_cast<VkVideoCodecOperationFlagBitsKHR>(0);
    result->vulkan_->qf[1].idx = static_cast<int>(context.decodeQueueFamily);
    result->vulkan_->qf[1].num = 1;
    result->vulkan_->qf[1].flags =
        static_cast<VkQueueFlagBits>(context.decodeQueueFlags);
    result->vulkan_->qf[1].video_caps =
        static_cast<VkVideoCodecOperationFlagBitsKHR>(context.decodeVideoCaps);
    result->vulkan_->nb_qf = 2;
    result->enabledExtensions_ = context.enabledDeviceExtensions;
    result->vulkan_->enabled_dev_extensions = result->enabledExtensions_.data();
    result->vulkan_->nb_enabled_dev_extensions =
        static_cast<int>(result->enabledExtensions_.size());

    const int initResult = av_hwdevice_ctx_init(result->ref_);
    if (initResult < 0) {
        result.reset();
        throw std::runtime_error(
            "av_hwdevice_ctx_init(vulkan) failed: " + AvErrorString(initResult));
    }
    return result;
}

VulkanVideoDevice::~VulkanVideoDevice() {
    if (ref_ != nullptr) {
        av_buffer_unref(&ref_);
    }
    vulkan_ = nullptr;
}

void VulkanVideoReader::Open(
    const std::string& path,
    const std::shared_ptr<VulkanVideoDevice>& device) {
    if (device == nullptr) {
        throw std::runtime_error("Vulkan Video device is not initialized");
    }
    path_ = path;
    device_ = device;

    int result = avformat_open_input(&format_, path.c_str(), nullptr, nullptr);
    CheckAv("avformat_open_input", result);
    result = avformat_find_stream_info(format_, nullptr);
    CheckAv("avformat_find_stream_info", result);
    streamIndex_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    CheckAv("av_find_best_stream", streamIndex_);
    streamTimeBase_ = format_->streams[streamIndex_]->time_base;

    const AVCodecParameters* parameters = format_->streams[streamIndex_]->codecpar;
    const AVCodec* decoder = avcodec_find_decoder(parameters->codec_id);
    if (decoder == nullptr) {
        throw std::runtime_error("FFmpeg Vulkan Video decoder is not enabled for " + path);
    }
    codec_ = avcodec_alloc_context3(decoder);
    if (codec_ == nullptr) {
        throw std::runtime_error("avcodec_alloc_context3 failed");
    }
    CheckAv("avcodec_parameters_to_context", avcodec_parameters_to_context(codec_, parameters));
    codec_->get_format = SelectVulkanFormat;
    codec_->hw_device_ctx = av_buffer_ref(device_->ref());
    if (codec_->hw_device_ctx == nullptr) {
        throw std::runtime_error("av_buffer_ref(vulkan device) failed");
    }
    CheckAv("avcodec_open2", avcodec_open2(codec_, decoder, nullptr));

    packet_ = av_packet_alloc();
    frame_ = av_frame_alloc();
    if (packet_ == nullptr || frame_ == nullptr) {
        throw std::runtime_error("FFmpeg packet/frame allocation failed");
    }
}

VulkanVideoReader::~VulkanVideoReader() {
    av_frame_free(&frame_);
    av_packet_free(&packet_);
    avcodec_free_context(&codec_);
    avformat_close_input(&format_);
}

void VulkanVideoReader::ThrowAvError(const char* operation, int errorCode) const {
    throw std::runtime_error(
        std::string(operation) + " failed for " + path_ + ": " + AvErrorString(errorCode));
}

bool VulkanVideoReader::Next(VulkanVideoFrame& output) {
    if (codec_ == nullptr || format_ == nullptr) {
        throw std::runtime_error("video reader was not opened");
    }
    av_frame_unref(frame_);

    for (;;) {
        const int receiveResult = avcodec_receive_frame(codec_, frame_);
        if (receiveResult == 0) {
            if (frame_->format != AV_PIX_FMT_VULKAN || frame_->data[0] == nullptr ||
                frame_->hw_frames_ctx == nullptr) {
                throw std::runtime_error(
                    "FFmpeg did not produce an AV_PIX_FMT_VULKAN frame for " + path_);
            }
            auto* framesContext = reinterpret_cast<AVHWFramesContext*>(frame_->hw_frames_ctx->data);
            auto* vkFrame = reinterpret_cast<AVVkFrame*>(frame_->data[0]);
            if (vkFrame == nullptr || vkFrame->img[0] == VK_NULL_HANDLE) {
                throw std::runtime_error("FFmpeg produced an empty Vulkan frame for " + path_);
            }
            output = {
                .frame = frame_,
                .vkFrame = vkFrame,
                .framesContext = framesContext,
                .width = static_cast<std::uint32_t>(frame_->width),
                .height = static_cast<std::uint32_t>(frame_->height),
                .timestampSeconds = frame_->best_effort_timestamp == AV_NOPTS_VALUE
                                        ? 0.0
                                        : frame_->best_effort_timestamp * av_q2d(streamTimeBase_),
                .softwareFormat = framesContext->sw_format,
                .colorRange = frame_->color_range,
                .colorSpace = frame_->colorspace,
                .codecFormat = static_cast<AVPixelFormat>(frame_->format),
            };
            return true;
        }
        if (receiveResult == AVERROR_EOF) {
            decoderEof_ = true;
            return false;
        }
        if (receiveResult != AVERROR(EAGAIN)) {
            ThrowAvError("avcodec_receive_frame", receiveResult);
        }

        if (inputEof_) {
            if (decoderEof_) {
                return false;
            }
            const int sendResult = avcodec_send_packet(codec_, nullptr);
            if (sendResult < 0 && sendResult != AVERROR(EAGAIN)) {
                ThrowAvError("avcodec_send_packet(eof)", sendResult);
            }
            continue;
        }

        for (;;) {
            const int readResult = av_read_frame(format_, packet_);
            if (readResult == AVERROR_EOF) {
                inputEof_ = true;
                break;
            }
            if (readResult < 0) {
                ThrowAvError("av_read_frame", readResult);
            }
            if (packet_->stream_index != streamIndex_) {
                av_packet_unref(packet_);
                continue;
            }
            const int sendResult = avcodec_send_packet(codec_, packet_);
            av_packet_unref(packet_);
            if (sendResult < 0 && sendResult != AVERROR(EAGAIN)) {
                ThrowAvError("avcodec_send_packet", sendResult);
            }
            break;
        }
    }
}
