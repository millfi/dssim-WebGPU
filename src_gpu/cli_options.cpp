#include "cli_options.h"

#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

bool IsValueOption(
    const std::string& argument,
    const std::string_view optionName) {
    return argument == optionName ||
           (argument.size() > optionName.size() &&
            argument.compare(0, optionName.size(), optionName) == 0 &&
            argument[optionName.size()] == '=');
}

std::string ReadOptionValue(
    const std::string& argument,
    const std::string_view optionName,
    int& argumentIndex,
    const int argumentCount,
    char** arguments) {
    if (argument == optionName) {
        if (argumentIndex + 1 >= argumentCount) {
            throw std::runtime_error(
                "missing value for " + std::string(optionName));
        }
        return arguments[++argumentIndex];
    }
    return argument.substr(optionName.size() + 1);
}

std::size_t ParsePipelineDepth(const std::string& value) {
    try {
        std::size_t parsedChars = 0;
        const unsigned long long parsed = std::stoull(value, &parsedChars);
        if (parsedChars != value.size() || parsed == 0u ||
            parsed > static_cast<unsigned long long>(
                         std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("invalid value");
        }
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "--pipeline-depth must be a positive integer");
    }
}

}  // namespace

CliOptions ParseArgs(int argc, char** argv) {
    CliOptions options;
    int positionalCount = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--stdin-pairs") {
            options.stdinPairsMode = true;
            continue;
        }
        if (arg == "--profiling") {
            options.profilingEnabled = true;
            continue;
        }

        if (IsValueOption(arg, "--pipeline-depth")) {
            options.pipelineDepth = ParsePipelineDepth(
                ReadOptionValue(
                    arg,
                    "--pipeline-depth",
                    i,
                    argc,
                    argv));
            options.pipelineDepthExplicit = true;
            continue;
        }

        if (IsValueOption(arg, "--out")) {
            options.out = ReadOptionValue(arg, "--out", i, argc, argv);
            continue;
        }

        if (IsValueOption(arg, "--csv")) {
            options.csv = ReadOptionValue(arg, "--csv", i, argc, argv);
            options.csvEnabled = true;
            continue;
        }

        if (IsValueOption(arg, "--debug-dump-dir")) {
            options.debugDumpDir = ReadOptionValue(
                arg,
                "--debug-dump-dir",
                i,
                argc,
                argv);
            options.debugDumpEnabled = true;
            continue;
        }

        if (!arg.empty() && arg[0] != '-') {
            if (positionalCount == 0) {
                options.image1 = arg;
            } else if (positionalCount == 1) {
                options.image2 = arg;
            } else {
                throw std::runtime_error("too many positional arguments");
            }
            ++positionalCount;
            continue;
        }

        throw std::runtime_error("unknown argument: " + arg);
    }

    if (options.debugDumpEnabled && options.debugDumpDir.empty()) {
        throw std::runtime_error("empty --debug-dump-dir");
    }
    if (options.csvEnabled && options.csv.empty()) {
        throw std::runtime_error("empty --csv path");
    }
    if (options.stdinPairsMode) {
        if (positionalCount != 0) {
            throw std::runtime_error(
                "--stdin-pairs does not accept positional image arguments");
        }
        if (!options.out.empty()) {
            throw std::runtime_error(
                "--stdin-pairs cannot be combined with --out");
        }
        if (options.csvEnabled) {
            throw std::runtime_error(
                "--stdin-pairs cannot be combined with --csv");
        }
        if (options.debugDumpEnabled) {
            throw std::runtime_error(
                "--stdin-pairs cannot be combined with --debug-dump-dir");
        }
        if (options.pipelineDepthExplicit) {
            throw std::runtime_error(
                "--stdin-pairs cannot be combined with --pipeline-depth");
        }
    } else if (positionalCount != 2) {
        throw std::runtime_error(
            "usage: dssim-WebGPU <img1> <img2> [--out <json>] "
            "[--csv <path>] [--pipeline-depth <N>] [--debug-dump-dir <dir>] "
            "[--stdin-pairs] [--profiling]");
    }

    return options;
}

ComparisonRequest ParseComparisonRequestLine(const std::string& line) {
    const std::size_t separator = line.find('\t');
    if (separator == std::string::npos) {
        throw std::runtime_error(
            "stdin pair line must be tab-delimited: <img1>\\t<img2>");
    }
    if (separator == 0 || separator + 1 >= line.size()) {
        throw std::runtime_error(
            "stdin pair line contains an empty image path");
    }
    return {
        .image1 = line.substr(0, separator),
        .image2 = line.substr(separator + 1),
    };
}
