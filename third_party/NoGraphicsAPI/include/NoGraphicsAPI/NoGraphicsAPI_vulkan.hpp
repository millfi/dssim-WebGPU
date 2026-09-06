#pragma once
#include <NoGraphicsAPI/NoGraphicsAPI.hpp>
#include <vulkan/vulkan.h>

namespace gpu {
// Local DSSIM extension. The callback may add queues, extensions, and features,
// preserving every requirement already present. Referenced storage must remain
// alive until create_device returns; NoGraphicsAPI owns the resulting device.
struct VulkanDeviceConfig {
    void* context = nullptr;
    VkResult (*configure)(void*, VkPhysicalDevice, VkDeviceCreateInfo&) = nullptr;
};
struct VulkanDevice {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
};
// Borrowed handles: never destroy them or submit the borrowed command buffer.
VulkanDevice get_vulkan_device(Device* device) noexcept;
VkCommandBuffer get_vulkan_commands(CommandBuffer* commands) noexcept;
// Cancel a failed headless recording without executing external-image work.
// Requires no pending newly created textures.
void discard_vulkan_commands(CommandBuffer* commands) noexcept;
// Native timeline points (e.g. FFmpeg frames) join the same submission.
void submit_vulkan(Span<CommandBuffer* const> commands, TimelinePoint completion,
                   Span<const VkSemaphoreSubmitInfo> waits,
                   Span<const VkSemaphoreSubmitInfo> signals) noexcept;
} // namespace gpu
