// Called by the Python integration test with independently exported GGUF fixtures.
#include "include/gguf_tokenizer.h"
#include "include/tonizer.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        Require(argc == 3 || argc == 4, "usage: compare <model.gguf> <export-dir> [--missing-byte|--pad]");
        auto direct = yan_lamma::GetGgufTokenizer(argv[1]);
        const std::string root = argv[2];
        auto exported = yan_lamma::GetTonizer(root + "/vocab.json", root + "/merges.txt",
                                             root + "/special_tokens.json");
        Require(direct != nullptr, "direct loader returned null");
        Require(direct->GetVocabSize() == exported->GetVocabSize(), "vocabulary size mismatch");
        Require(direct->GetBOSId() == exported->GetBOSId() &&
                direct->GetEOSId() == exported->GetEOSId() &&
                direct->GetUNKId() == exported->GetUNKId(), "special IDs mismatch");
        if (argc == 4 && std::string(argv[3]) == "--missing-byte") {
            for (const yan_lamma::Tonizer* tokenizer :
                 std::vector<const yan_lamma::Tonizer*>{direct.get(), exported.get()}) {
                bool failed = false;
                try { (void)tokenizer->Encode(std::string(1, '\xFF')); }
                catch (const std::runtime_error&) { failed = true; }
                Require(failed, "missing UNK silently emitted an invalid ID");
            }
            return 0;
        }
        for (int id = 0; id < direct->GetVocabSize(); ++id) {
            Require(direct->DecodeToken(id) == exported->DecodeToken(id), "token decode mismatch");
        }
        const std::size_t markers = (direct->GetBOSId() >= 0) + (direct->GetEOSId() >= 0);
        if (argc == 4) {
            Require(std::string(argv[3]) == "--pad", "unknown test mode");
            Require(direct->Encode(u8"<填充>").size() == markers + 1, "GGUF PAD was split");
            Require(exported->Encode(u8"<填充>").size() == markers + 1, "BPE PAD was split");
        }
        Require(direct->Encode("<ctrl>").size() == markers + 1, "control token was split");
        Require(direct->Encode("<ctrl>long").size() == markers + 1, "longest control token was split");
        Require(exported->Encode("<|literal|>").size() > markers + 1,
                "explicit special_ids did not replace legacy inference");
        std::string bytes;
        for (int b = 0; b < 256; ++b) bytes += static_cast<char>(b);
        for (const auto& text : std::vector<std::string>{"", "hello abc aaa", " \t\r\n", u8"中文😀",
                 std::string("a\0b", 3), bytes, "a<ctrl>b", "<ctrl>long<ctrl>", u8"<控制>",
                 u8"<填充>", "<|literal|>"}) {
            const auto ids = direct->Encode(text);
            Require(ids == exported->Encode(text), "encode IDs mismatch");
            std::string expected = text;
            if (direct->GetBOSId() >= 0) expected.insert(0, direct->DecodeToken(direct->GetBOSId()));
            if (direct->GetEOSId() >= 0) expected += direct->DecodeToken(direct->GetEOSId());
            Require(direct->Decode(ids) == expected, "direct roundtrip mismatch");
            Require(exported->Decode(ids) == expected, "exported roundtrip mismatch");
        }
        std::cout << "GGUF and exported BPE paths agree\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
