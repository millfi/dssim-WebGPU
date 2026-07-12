#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/pixfmt.h>
}

struct VulkanInteropContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t computeQueueFamily = 0;
    std::uint32_t decodeQueueFamily = VK_QUEUE_FAMILY_IGNORED;
    VkQueueFlags computeQueueFlags = 0;
    VkQueueFlags decodeQueueFlags = 0;
    VkVideoCodecOperationFlagsKHR decodeVideoCaps = 0;
    std::vector<const char*> enabledDeviceExtensions;
};

struct VulkanVideoFrame {
    AVFrame* frame = nullptr;
    AVVkFrame* vkFrame = nullptr;
    AVHWFramesContext* framesContext = nullptr;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    double timestampSeconds = 0.0;
    AVPixelFormat softwareFormat = AV_PIX_FMT_NONE;
    AVColorRange colorRange = AVCOL_RANGE_UNSPECIFIED;
    AVColorSpace colorSpace = AVCOL_SPC_UNSPECIFIED;
    AVPixelFormat codecFormat = AV_PIX_FMT_NONE;
};

class VulkanVideoDevice;

class VulkanVideoReader {
public:
    VulkanVideoReader() = default;
    ~VulkanVideoReader();
    VulkanVideoReader(const VulkanVideoReader&) = delete;
    VulkanVideoReader& operator=(const VulkanVideoReader&) = delete;

    void Open(const std::string& path, const std::shared_ptr<VulkanVideoDevice>& device);
    bool Next(VulkanVideoFrame& output);
    const std::string& path() const noexcept { return path_; }

private:
    void DrainPackets();
    void ThrowAvError(const char* operation, int errorCode) const;

    std::string path_;
    std::shared_ptr<VulkanVideoDevice> device_;
    AVFormatContext* format_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVPacket* packet_ = nullptr;
    AVFrame* frame_ = nullptr;
    int streamIndex_ = -1;
    AVRational streamTimeBase_{};
    bool inputEof_ = false;
    bool decoderEof_ = false;
    bool packetPending_ = false;
};

class VulkanVideoDevice : public std::enable_shared_from_this<VulkanVideoDevice> {
public:
    static std::shared_ptr<VulkanVideoDevice> Create(const VulkanInteropContext& context);
    ~VulkanVideoDevice();

    VulkanVideoDevice(const VulkanVideoDevice&) = delete;
    VulkanVideoDevice& operator=(const VulkanVideoDevice&) = delete;

    AVBufferRef* ref() const noexcept { return ref_; }
    AVVulkanDeviceContext* vulkan() const noexcept { return vulkan_; }

private:
    VulkanVideoDevice() = default;
    AVBufferRef* ref_ = nullptr;
    AVVulkanDeviceContext* vulkan_ = nullptr;
    std::vector<const char*> enabledExtensions_;
};

bool IsVideoPath(const std::string& path);
std::string AvErrorString(int errorCode);
AVCodecID ProbeVideoCodec(const std::string& path);
VkVideoCodecOperationFlagsKHR VulkanVideoCodecOperationForCodec(AVCodecID codecId);
