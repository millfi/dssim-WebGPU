#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

struct DecodedImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    std::vector<std::uint8_t> pixels;
};

DecodedImage LoadPngRgba8(const std::filesystem::path& path);
