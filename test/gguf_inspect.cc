// Inspect GGUF metadata and tensor descriptors without loading model weights.
#include "include/gguf.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Usage(std::ostream& out) {
    out << "usage: gguf_inspect <model.gguf>\n"
           "  display the GGUF header, metadata, and tensor descriptors\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        Usage(std::cout);
        return 0;
    }
    if (argc != 2) {
        Usage(std::cerr);
        return 1;
    }

    const char* stage = "GGUF load";
    try {
        const std::string path = argv[1];
        yan_lamma::GgufRead reader;
        if (!reader.load(path)) {
            throw std::runtime_error(reader.load_error_.empty()
                ? "failed to load GGUF file: " + path
                : reader.load_error_);
        }

        stage = "inspect";
        if (!yan_lamma::Inspect(reader)) {
            throw std::runtime_error("failed to inspect GGUF file: " + path);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL [" << stage << "]: " << error.what() << '\n';
        return 1;
    }
}
