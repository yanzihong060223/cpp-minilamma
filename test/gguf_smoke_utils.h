#pragma once

// Validation shared by the manual GGUF runners.
#include "include/tensor.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace yan_lamma::smoke {

inline void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline void CheckLogits(const Tensor& logits, int vocab_size) {
    Require(logits.shape == std::vector<int>{vocab_size} &&
                logits.data.size() == static_cast<std::size_t>(vocab_size),
            "logits shape/storage does not match vocabulary");
    for (float value : logits.data) {
        Require(std::isfinite(value), "logits contain NaN/Inf");
    }
}

}  // namespace yan_lamma::smoke
