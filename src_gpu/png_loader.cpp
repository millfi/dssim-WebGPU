#include "png_loader.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <png.h>

#if defined(_MSC_VER)
#pragma warning(disable : 4611)
#endif

namespace {

void CloseFile(FILE* fp) {
    if (fp != nullptr) {
        std::fclose(fp);
    }
}

}  // namespace

DecodedImage LoadPngRgba8(const std::filesystem::path& path) {
    FILE* fp = nullptr;
#if defined(_WIN32)
    if (_wfopen_s(&fp, path.c_str(), L"rb") != 0 || fp == nullptr) {
        throw std::runtime_error("failed to open png: " + path.string());
    }
#else
    fp = std::fopen(path.string().c_str(), "rb");
    if (fp == nullptr) {
        throw std::runtime_error("failed to open png: " + path.string());
    }
#endif

    std::array<unsigned char, 8> sig = {};
    if (std::fread(sig.data(), 1, sig.size(), fp) != sig.size() ||
        png_sig_cmp(sig.data(), 0, sig.size()) != 0) {
        CloseFile(fp);
        throw std::runtime_error("not a valid png file: " + path.string());
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png == nullptr) {
        CloseFile(fp);
        throw std::runtime_error("png_create_read_struct failed");
    }
    png_infop info = png_create_info_struct(png);
    png_infop endInfo = png_create_info_struct(png);
    if (info == nullptr || endInfo == nullptr) {
        png_destroy_read_struct(&png, &info, &endInfo);
        CloseFile(fp);
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png)) != 0) {
        png_destroy_read_struct(&png, &info, &endInfo);
        CloseFile(fp);
        throw std::runtime_error("libpng decode failure: " + path.string());
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, static_cast<int>(sig.size()));
    png_read_info(png, info);

    const png_uint_32 width = png_get_image_width(png, info);
    const png_uint_32 height = png_get_image_height(png, info);
    const int bitDepth = png_get_bit_depth(png, info);
    const int colorType = png_get_color_type(png, info);

    if (width == 0 || height == 0) {
        png_destroy_read_struct(&png, &info, &endInfo);
        CloseFile(fp);
        throw std::runtime_error("png has zero dimensions: " + path.string());
    }

    if (bitDepth == 16) {
        png_set_strip_16(png);
    }
    if (colorType == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }
    if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }
    if (png_get_valid(png, info, PNG_INFO_tRNS) != 0) {
        png_set_tRNS_to_alpha(png);
    }
    if ((colorType & PNG_COLOR_MASK_ALPHA) == 0) {
        png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER);
    }
    if (colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);
    const std::size_t rowBytes = png_get_rowbytes(png, info);
    const int outChannels = png_get_channels(png, info);

    if (outChannels != 4) {
        png_destroy_read_struct(&png, &info, &endInfo);
        CloseFile(fp);
        throw std::runtime_error("unexpected decoded channel count: " + std::to_string(outChannels));
    }

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(rowBytes) * height);
    std::vector<png_bytep> rows(height);
    for (png_uint_32 y = 0; y < height; ++y) {
        rows[y] = pixels.data() + static_cast<std::size_t>(y) * rowBytes;
    }
    png_read_image(png, rows.data());
    png_read_end(png, endInfo);

    png_destroy_read_struct(&png, &info, &endInfo);
    CloseFile(fp);

    DecodedImage out;
    out.width = width;
    out.height = height;
    out.channels = static_cast<std::uint32_t>(outChannels);
    out.pixels = std::move(pixels);
    return out;
}
