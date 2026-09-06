#define VK_ENABLE_BETA_EXTENSIONS

#include <vulkan/vulkan.h>

extern "C" {
#include <libavcodec/codec_id.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace {

struct DeviceState {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::vector<const char*> extensions;
};

VkVideoCodecOperationFlagBitsKHR CodecOperation(AVCodecID codec) {
    switch (codec) {
        case AV_CODEC_ID_H264:
            return VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
        case AV_CODEC_ID_HEVC:
            return VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
        case AV_CODEC_ID_VP9:
            return VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
        case AV_CODEC_ID_AV1:
            return VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
        default:
            return static_cast<VkVideoCodecOperationFlagBitsKHR>(0);
    }
}

bool HasExtension(VkPhysicalDevice device, const char* name) {
    std::uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()) != VK_SUCCESS) {
        return false;
    }
    return std::any_of(
        extensions.begin(), extensions.end(),
        [name](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, name) == 0;
        });
}

void FreeDevice(AVHWDeviceContext* context) {
    auto* state = static_cast<DeviceState*>(context->user_opaque);
    if (state != nullptr) {
        if (state->device != VK_NULL_HANDLE) {
            vkDestroyDevice(state->device, nullptr);
        }
        if (state->instance != VK_NULL_HANDLE) {
            vkDestroyInstance(state->instance, nullptr);
        }
        delete state;
    }
}

}  // namespace

extern "C" AVBufferRef* dssim_create_vulkan_device(AVCodecID codec) {
    const VkVideoCodecOperationFlagBitsKHR operation = CodecOperation(codec);
    if (operation == 0) {
        return nullptr;
    }

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "dssim-reference";
    applicationInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &applicationInfo;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
        return nullptr;
    }

    std::uint32_t deviceCount = 0;
    if (vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr) != VK_SUCCESS || deviceCount == 0) {
        vkDestroyInstance(instance, nullptr);
        return nullptr;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    if (vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return nullptr;
    }

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    std::uint32_t computeQueueFamily = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED;
    VkQueueFlags computeQueueFlags = 0;
    VkQueueFlags queueFlags = 0;
    bool hasMaintenance1 = false;
    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (VK_API_VERSION_MAJOR(properties.apiVersion) < 1 ||
            (VK_API_VERSION_MAJOR(properties.apiVersion) == 1 &&
             VK_API_VERSION_MINOR(properties.apiVersion) < 3)) {
            continue;
        }
        const char* codecExtension =
            operation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR ? VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME :
            operation == VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR ? VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME :
            operation == VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR ? VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME :
            VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME;
        if (!HasExtension(candidate, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) ||
            !HasExtension(candidate, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME) ||
            !HasExtension(candidate, codecExtension)) {
            continue;
        }

        std::uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties2(candidate, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties2> queues(queueCount);
        std::vector<VkQueueFamilyVideoPropertiesKHR> videoProperties(queueCount);
        for (std::uint32_t i = 0; i < queueCount; ++i) {
            queues[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
            videoProperties[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
            queues[i].pNext = &videoProperties[i];
        }
        vkGetPhysicalDeviceQueueFamilyProperties2(candidate, &queueCount, queues.data());
        std::uint32_t candidateComputeQueueFamily = VK_QUEUE_FAMILY_IGNORED;
        VkQueueFlags candidateComputeQueueFlags = 0;
        for (std::uint32_t i = 0; i < queueCount; ++i) {
            if (queues[i].queueFamilyProperties.queueCount != 0 &&
                (queues[i].queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                candidateComputeQueueFamily = i;
                candidateComputeQueueFlags = queues[i].queueFamilyProperties.queueFlags;
                break;
            }
        }
        if (candidateComputeQueueFamily == VK_QUEUE_FAMILY_IGNORED) {
            continue;
        }
        for (std::uint32_t i = 0; i < queueCount; ++i) {
            if ((queues[i].queueFamilyProperties.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) != 0 &&
                (videoProperties[i].videoCodecOperations & operation) != 0) {
                physicalDevice = candidate;
                computeQueueFamily = candidateComputeQueueFamily;
                queueFamily = i;
                computeQueueFlags = candidateComputeQueueFlags;
                queueFlags = queues[i].queueFamilyProperties.queueFlags;
                hasMaintenance1 = HasExtension(candidate, VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);
                break;
            }
        }
        if (physicalDevice != VK_NULL_HANDLE) {
            break;
        }
    }
    if (physicalDevice == VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        return nullptr;
    }

    auto state = std::make_unique<DeviceState>();
    state->instance = instance;
    state->extensions = {
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
    };
    const char* codecExtension =
        operation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR ? VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME :
        operation == VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR ? VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME :
        operation == VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR ? VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME :
        VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME;
    state->extensions.push_back(codecExtension);
    if (hasMaintenance1) {
        state->extensions.push_back(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);
    }

    VkPhysicalDeviceVulkan13Features vulkan13{};
    vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceVulkan12Features vulkan12{};
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12.pNext = &vulkan13;
    VkPhysicalDeviceVideoMaintenance1FeaturesKHR maintenance1{};
    maintenance1.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR;
    if (hasMaintenance1) {
        vulkan13.pNext = &maintenance1;
    }
    VkPhysicalDeviceFeatures2 available{};
    available.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    available.pNext = &vulkan12;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &available);
    vulkan12.timelineSemaphore = VK_TRUE;
    vulkan13.synchronization2 = VK_TRUE;
    vulkan13.dynamicRendering = VK_TRUE;
    if (hasMaintenance1) {
        maintenance1.videoMaintenance1 = VK_TRUE;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfos[2]{};
    queueInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfos[0].queueFamilyIndex = computeQueueFamily;
    queueInfos[0].queueCount = 1;
    queueInfos[0].pQueuePriorities = &priority;
    std::uint32_t queueInfoCount = 1;
    if (queueFamily != computeQueueFamily) {
        queueInfos[queueInfoCount].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfos[queueInfoCount].queueFamilyIndex = queueFamily;
        queueInfos[queueInfoCount].queueCount = 1;
        queueInfos[queueInfoCount].pQueuePriorities = &priority;
        ++queueInfoCount;
    }
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &vulkan12;
    deviceInfo.queueCreateInfoCount = queueInfoCount;
    deviceInfo.pQueueCreateInfos = queueInfos;
    deviceInfo.enabledExtensionCount = static_cast<std::uint32_t>(state->extensions.size());
    deviceInfo.ppEnabledExtensionNames = state->extensions.data();
    if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &state->device) != VK_SUCCESS) {
        state->instance = VK_NULL_HANDLE;
        vkDestroyInstance(instance, nullptr);
        return nullptr;
    }

    AVBufferRef* reference = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
    if (reference == nullptr) {
        vkDestroyDevice(state->device, nullptr);
        vkDestroyInstance(state->instance, nullptr);
        return nullptr;
    }
    auto* context = reinterpret_cast<AVHWDeviceContext*>(reference->data);
    auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(context->hwctx);
    vulkan->get_proc_addr = vkGetInstanceProcAddr;
    vulkan->inst = instance;
    vulkan->phys_dev = physicalDevice;
    vulkan->act_dev = state->device;
    vulkan->qf[0].idx = static_cast<int>(computeQueueFamily);
    vulkan->qf[0].num = 1;
    vulkan->qf[0].flags = static_cast<VkQueueFlagBits>(computeQueueFlags);
    vulkan->qf[0].video_caps = static_cast<VkVideoCodecOperationFlagBitsKHR>(0);
    vulkan->qf[1].idx = static_cast<int>(queueFamily);
    vulkan->qf[1].num = 1;
    vulkan->qf[1].flags = static_cast<VkQueueFlagBits>(queueFlags);
    vulkan->qf[1].video_caps = operation;
    vulkan->nb_qf = 2;
    vulkan->enabled_dev_extensions = state->extensions.data();
    vulkan->nb_enabled_dev_extensions = static_cast<int>(state->extensions.size());
    context->user_opaque = state.release();
    context->free = FreeDevice;
    const int result = av_hwdevice_ctx_init(reference);
    if (result < 0) {
        av_buffer_unref(&reference);
        return nullptr;
    }
    return reference;
}
