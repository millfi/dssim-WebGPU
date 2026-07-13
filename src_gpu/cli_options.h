#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

struct CliOptions {
    std::filesystem::path image1;
    std::filesystem::path image2;
    std::filesystem::path out;
    std::filesystem::path csv;
    std::filesystem::path debugDumpDir;
    bool debugDumpEnabled = false;
    bool stdinPairsMode = false;
    bool profilingEnabled = false;
    bool csvEnabled = false;
    std::size_t pipelineDepth = 3;
    bool pipelineDepthExplicit = false;
};

struct ComparisonRequest {
    std::filesystem::path image1;
    std::filesystem::path image2;
};

CliOptions ParseArgs(int argc, char** argv);
ComparisonRequest ParseComparisonRequestLine(const std::string& line);
