#include "image_loader.h"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

namespace {

std::string AvErrorString(int errorCode) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> buffer{};
    av_strerror(errorCode, buffer.data(), buffer.size());
    return std::string(buffer.data());
}

void CheckAv(const char* operation, int errorCode, const std::string& path) {
    if (errorCode < 0) {
        throw std::runtime_error(
            std::string(operation) + " failed for " + path + ": " + AvErrorString(errorCode));
    }
}

enum AVPixelFormat SelectSoftwareFormat(
    AVCodecContext*, const enum AVPixelFormat* formats) {
    for (const enum AVPixelFormat* current = formats;
         current != nullptr && *current != AV_PIX_FMT_NONE;
         ++current) {
        const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(*current);
        if (descriptor != nullptr && (descriptor->flags & AV_PIX_FMT_FLAG_HWACCEL) == 0) {
            return *current;
        }
    }
    return AV_PIX_FMT_NONE;
}

}  // namespace

DecodedImage LoadImageRgba8(const std::filesystem::path& path) {
    const std::string pathString = path.string();
    AVFormatContext* format = nullptr;
    AVCodecContext* codec = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    SwsContext* scaler = nullptr;

    const auto cleanup = [&]() {
        sws_freeContext(scaler);
        scaler = nullptr;
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec);
        avformat_close_input(&format);
    };

    try {
        CheckAv("avformat_open_input", avformat_open_input(&format, pathString.c_str(), nullptr, nullptr), pathString);
        CheckAv("avformat_find_stream_info", avformat_find_stream_info(format, nullptr), pathString);

        const int streamIndex = av_find_best_stream(
            format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        CheckAv("av_find_best_stream", streamIndex, pathString);
        const AVCodecParameters* parameters = format->streams[streamIndex]->codecpar;
        const AVCodec* decoder = avcodec_find_decoder(parameters->codec_id);
        if (decoder == nullptr) {
            throw std::runtime_error(
                "FFmpeg image decoder is not enabled for " + pathString +
                " (codec " + avcodec_get_name(parameters->codec_id) + ")");
        }

        codec = avcodec_alloc_context3(decoder);
        if (codec == nullptr) {
            throw std::runtime_error("avcodec_alloc_context3 failed for " + pathString);
        }
        CheckAv("avcodec_parameters_to_context", avcodec_parameters_to_context(codec, parameters), pathString);
        codec->get_format = SelectSoftwareFormat;
        CheckAv("avcodec_open2", avcodec_open2(codec, decoder, nullptr), pathString);

        packet = av_packet_alloc();
        frame = av_frame_alloc();
        if (packet == nullptr || frame == nullptr) {
            throw std::runtime_error("FFmpeg packet/frame allocation failed for " + pathString);
        }

        bool inputEof = false;
        for (;;) {
            int result = avcodec_receive_frame(codec, frame);
            if (result == 0) {
                break;
            }
            if (result == AVERROR_EOF) {
                throw std::runtime_error("FFmpeg produced no image frame for " + pathString);
            }
            if (result != AVERROR(EAGAIN)) {
                CheckAv("avcodec_receive_frame", result, pathString);
            }

            if (inputEof) {
                result = avcodec_send_packet(codec, nullptr);
                if (result < 0 && result != AVERROR(EAGAIN) && result != AVERROR_EOF) {
                    CheckAv("avcodec_send_packet(eof)", result, pathString);
                }
                continue;
            }

            for (;;) {
                result = av_read_frame(format, packet);
                if (result == AVERROR_EOF) {
                    inputEof = true;
                    break;
                }
                if (result < 0) {
                    CheckAv("av_read_frame", result, pathString);
                }
                if (packet->stream_index != streamIndex) {
                    av_packet_unref(packet);
                    continue;
                }
                result = avcodec_send_packet(codec, packet);
                av_packet_unref(packet);
                if (result < 0 && result != AVERROR(EAGAIN)) {
                    CheckAv("avcodec_send_packet", result, pathString);
                }
                break;
            }
        }

        if (frame->width <= 0 || frame->height <= 0 || frame->format == AV_PIX_FMT_NONE) {
            throw std::runtime_error("FFmpeg produced an invalid image frame for " + pathString);
        }

        scaler = sws_getContext(
            frame->width,
            frame->height,
            static_cast<AVPixelFormat>(frame->format),
            frame->width,
            frame->height,
            AV_PIX_FMT_RGBA,
            SWS_BILINEAR,
            nullptr,
            nullptr,
            nullptr);
        if (scaler == nullptr) {
            throw std::runtime_error("sws_getContext failed for " + pathString);
        }

        const int bufferSize = av_image_get_buffer_size(
            AV_PIX_FMT_RGBA, frame->width, frame->height, 1);
        CheckAv("av_image_get_buffer_size", bufferSize, pathString);
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(bufferSize));
        std::array<std::uint8_t*, 4> destinationData{};
        std::array<int, 4> destinationLinesize{};
        CheckAv(
            "av_image_fill_arrays",
            av_image_fill_arrays(
                destinationData.data(),
                destinationLinesize.data(),
                pixels.data(),
                AV_PIX_FMT_RGBA,
                frame->width,
                frame->height,
                1),
            pathString);

        const int scaledHeight = sws_scale(
            scaler,
            frame->data,
            frame->linesize,
            0,
            frame->height,
            destinationData.data(),
            destinationLinesize.data());
        if (scaledHeight != frame->height) {
            throw std::runtime_error("sws_scale produced an incomplete image for " + pathString);
        }

        DecodedImage output;
        output.width = static_cast<std::uint32_t>(frame->width);
        output.height = static_cast<std::uint32_t>(frame->height);
        output.channels = 4;
        output.pixels = std::move(pixels);
        cleanup();
        return output;
    } catch (...) {
        cleanup();
        throw;
    }
}
