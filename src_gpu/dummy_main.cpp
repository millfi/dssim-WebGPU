#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path image1;
    std::filesystem::path image2;
    std::filesystem::path out;
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
                    os << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
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

CliOptions ParseArgs(int argc, char** argv) {
    if (argc < 5) {
        throw std::runtime_error("usage: dssim_gpu_dummy <img1> <img2> --out <json>");
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

        throw std::runtime_error("unknown argument: " + arg);
    }

    if (options.out.empty()) {
        throw std::runtime_error("missing --out <json>");
    }

    return options;
}

std::string BuildJson(
    const std::filesystem::path& image1,
    const std::filesystem::path& image2,
    const std::filesystem::path& out,
    double score,
    const std::string& scoreText) {
    const auto abs1 = std::filesystem::absolute(image1).string();
    const auto abs2 = std::filesystem::absolute(image2).string();
    const auto absOut = std::filesystem::absolute(out).string();
    const std::string command = "dssim_gpu_dummy \"" + abs1 + "\" \"" + abs2 + "\" --out \"" + absOut + "\"";

    std::ostringstream os;
    os << "{\n";
    os << "  \"schema_version\": 1,\n";
    os << "  \"engine\": \"gpu-dummy-cpp\",\n";
    os << "  \"status\": \"ok\",\n";
    os << "  \"input\": {\n";
    os << "    \"image1\": \"" << EscapeJson(abs1) << "\",\n";
    os << "    \"image2\": \"" << EscapeJson(abs2) << "\"\n";
    os << "  },\n";
    os << "  \"command\": \"" << EscapeJson(command) << "\",\n";
    os << "  \"version\": \"dummy-0\",\n";
    os << "  \"result\": {\n";
    os << "    \"score_text\": \"" << scoreText << "\",\n";
    os << "    \"score_f64\": " << std::setprecision(17) << score << ",\n";
    os << "    \"score_bits_u64\": \"" << ToHexU64(score) << "\",\n";
    os << "    \"compared_path\": \"" << EscapeJson(abs2) << "\"\n";
    os << "  }\n";
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = ParseArgs(argc, argv);
        const std::vector<std::uint8_t> bytes1 = ReadAllBytes(options.image1);
        const std::vector<std::uint8_t> bytes2 = ReadAllBytes(options.image2);

        std::uint64_t sum1 = 0;
        for (std::uint8_t value : bytes1) {
            sum1 += static_cast<std::uint64_t>(value);
        }

        std::uint64_t sum2 = 0;
        for (std::uint8_t value : bytes2) {
            sum2 += static_cast<std::uint64_t>(value);
        }

        const auto maxLen = static_cast<double>(std::max(bytes1.size(), bytes2.size()));
        const double denominator = std::max(1.0, maxLen * 255.0);
        const double score = std::abs(static_cast<double>(sum1) - static_cast<double>(sum2)) / denominator;

        std::ostringstream scoreText;
        scoreText << std::fixed << std::setprecision(8) << score;

        const std::string json = BuildJson(options.image1, options.image2, options.out, score, scoreText.str());
        WriteStringFile(options.out, json);

        std::cout << scoreText.str() << '\t' << options.image2.string() << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "dssim_gpu_dummy error: " << ex.what() << '\n';
        return 1;
    }
}
