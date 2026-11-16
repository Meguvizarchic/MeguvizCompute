#pragma once
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
/* RGBA8 import uses a typed image, deliberately separate from ncnn's external
 * YCbCr sampling allocator. Every feature is checked against this AHB. */
class AhbAllocator final : public ncnn::VkAllocator
{
public:
    AHardwareBuffer *buffer;
    VkAndroidHardwareBufferPropertiesANDROID properties{};
    VkAndroidHardwareBufferFormatPropertiesANDROID format{};
    bool valid = false;
    VkResult last_result = VK_SUCCESS;
    explicit AhbAllocator(const ncnn::VulkanDevice *device, AHardwareBuffer *ahb)
        : VkAllocator(device), buffer(ahb)
    {
        format.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID;
        properties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
        properties.pNext = &format;
        if (!device->vkGetAndroidHardwareBufferPropertiesANDROID) return;
        last_result = device->vkGetAndroidHardwareBufferPropertiesANDROID(device->vkdevice(), buffer, &properties);
        valid = last_result == VK_SUCCESS && format.format == VK_FORMAT_R8G8B8A8_UNORM &&
                (format.formatFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) &&
                (format.formatFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
    }
    ncnn::VkBufferMemory *fastMalloc(size_t) override { return nullptr; }
    void fastFree(ncnn::VkBufferMemory *) override {}
    ncnn::VkImageMemory *fastMalloc(int w, int h, int, size_t, int) override
    {
        if (!valid) return nullptr;
        auto p = std::make_unique<ncnn::VkImageMemory>();
        std::memset(p.get(), 0, sizeof(*p));
        VkExternalMemoryImageCreateInfo external{}; external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
        VkImageCreateInfo image{}; image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.pNext = &external;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = VK_FORMAT_R8G8B8A8_UNORM;
        image.extent = {uint32_t(w), uint32_t(h), 1};
        image.mipLevels = image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        last_result = ncnn::vkCreateImage(vkdev->vkdevice(), &image, nullptr, &p->image);
        if (last_result != VK_SUCCESS) return nullptr;
        VkImportAndroidHardwareBufferInfoANDROID import{}; import.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
        import.buffer = buffer;
        VkMemoryDedicatedAllocateInfo dedicated{}; dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicated.pNext = &import;
        dedicated.image = p->image;
        VkMemoryAllocateInfo memory{}; memory.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memory.pNext = &dedicated;
        memory.allocationSize = properties.allocationSize;
        memory.memoryTypeIndex = vkdev->find_memory_index(properties.memoryTypeBits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        if (memory.memoryTypeIndex == UINT32_MAX)
        { ncnn::vkDestroyImage(vkdev->vkdevice(), p->image, nullptr); return nullptr; }
        last_result = ncnn::vkAllocateMemory(vkdev->vkdevice(), &memory, nullptr, &p->memory);
        if (last_result != VK_SUCCESS)
        { ncnn::vkDestroyImage(vkdev->vkdevice(), p->image, nullptr); return nullptr; }
        last_result = ncnn::vkBindImageMemory(vkdev->vkdevice(), p->image, p->memory, 0);
        if (last_result != VK_SUCCESS)
        { ncnn::vkDestroyImage(vkdev->vkdevice(), p->image, nullptr); ncnn::vkFreeMemory(vkdev->vkdevice(), p->memory, nullptr); return nullptr; }
        VkImageViewCreateInfo view{}; view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = p->image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = image.format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        last_result = ncnn::vkCreateImageView(vkdev->vkdevice(), &view, nullptr, &p->imageview);
        if (last_result != VK_SUCCESS)
        { ncnn::vkDestroyImage(vkdev->vkdevice(), p->image, nullptr); ncnn::vkFreeMemory(vkdev->vkdevice(), p->memory, nullptr); return nullptr; }
        p->width = w; p->height = h; p->depth = 1; p->format = image.format;
        p->bind_capacity = properties.allocationSize;
        p->memory_type_index = memory.memoryTypeIndex;
        p->image_layout = VK_IMAGE_LAYOUT_GENERAL;
        p->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        return p.release();
    }
    void fastFree(ncnn::VkImageMemory *p) override
    {
        ncnn::vkDestroyImageView(vkdev->vkdevice(), p->imageview, nullptr);
        ncnn::vkDestroyImage(vkdev->vkdevice(), p->image, nullptr);
        ncnn::vkFreeMemory(vkdev->vkdevice(), p->memory, nullptr);
        delete p;
    }
};
#endif
