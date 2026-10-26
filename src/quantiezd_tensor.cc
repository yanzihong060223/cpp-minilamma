#include "include/quantized_tensor.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace yan_lamma {
namespace {
inline uint16_t ReadU16LE(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) |
           (static_cast<uint16_t>(p[1]) << 8);
}

// IEEE-754 half → float。覆盖 ±0 / subnormal / inf / NaN / normal。
inline float Fp16ToFp32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 0x1u;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t frac =  h        & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign << 31;
        } else {
            uint32_t e = 1;
            uint32_t m = frac;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FFu;
            out = (sign << 31) | ((e + (127u - 15u)) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        out = (sign << 31) | (0xFFu << 23) | (frac << 13);
    } else {
        out = (sign << 31) | ((exp + (127u - 15u)) << 23) | (frac << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}
}  // namespace

Tensor ToTensor(const  QuanTensor& q) {
    if (q.type !=  QuantType::kF32) {
        throw std::runtime_error(" ToTensor: expected F32 QuantizedTensor, got quantized type");
    }
    if (q.shape.empty() && q.kF32_data.empty()) {
        return Tensor();
    }
    if (q.GetNumSize() != q.kF32_data.size()) {
        throw std::runtime_error("ToTensor: shape and F32 data size mismatch");
    }
    Tensor t (q.shape);
    t.data = q.kF32_data;
    return t;

}
QuanTensor ToQuanTensor(const Tensor& t) {
     QuanTensor q;
     q.type = QuantType::kF32;
     q.shape = t.shape;
     if ((!t.shape.empty() || !t.data.empty()) &&
         q.GetNumSize() != t.data.size()) {
         throw std::runtime_error("ToQuanTensor: shape and data size mismatch");
     }
     q.kF32_data = t.data;
     return q;
}

Tensor Q80ToTensor(const QuanTensor& q) {
    if (q.type != QuantType::kQ80) {
        throw std::runtime_error("Q80ToTensor: expected Q8_0 QuantizedTensor");
    }
    if (q.shape.empty() && q.kQ80_data.empty()) {
        return Tensor();
    }
    const size_t num = q.GetNumSize();
    if (num % kQuantBlockSize != 0) {
        throw std::runtime_error("Q80ToTensor: element count not block-aligned");
    }
    const size_t n_blocks = num / kQuantBlockSize;
    if (q.kQ80_data.size() != n_blocks * kBytesPerQ80Block) {
        throw std::runtime_error("Q80ToTensor: data size mismatch");
    }
    Tensor t(q.shape);
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = q.kQ80_data.data() + b * kBytesPerQ80Block;
        const float    d   = Fp16ToFp32(ReadU16LE(blk));
        const int8_t*  qs  = reinterpret_cast<const int8_t*>(blk + 2);
        float* out = t.data.data() + b * kQuantBlockSize;
        for (size_t i = 0; i < kQuantBlockSize; ++i) {
            out[i] = d * static_cast<float>(qs[i]);
        }
    }
    return t;
}

Tensor Q40ToTensor(const QuanTensor& q) {
    if (q.type != QuantType::kQ40) {
        throw std::runtime_error("Q40ToTensor: expected Q4_0 QuantizedTensor");
    }
    if (q.shape.empty() && q.kQ40_data.empty()) {
        return Tensor();
    }
    const size_t num = q.GetNumSize();
    if (num % kQuantBlockSize != 0) {
        throw std::runtime_error("Q40ToTensor: element count not block-aligned");
    }
    const size_t n_blocks = num / kQuantBlockSize;
    if (q.kQ40_data.size() != n_blocks * kBytesPerQ40Block) {
        throw std::runtime_error("Q40ToTensor: data size mismatch");
    }
    Tensor t(q.shape);
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = q.kQ40_data.data() + b * kBytesPerQ40Block;
        const float    d   = Fp16ToFp32(ReadU16LE(blk));
        const uint8_t* qs  = blk + 2;
        float* out = t.data.data() + b * kQuantBlockSize;
        for (size_t i = 0; i < 16; ++i) {
            const int lo = static_cast<int>(qs[i] & 0x0F) - 8;
            const int hi = static_cast<int>(qs[i] >> 4)   - 8;
            out[i]      = d * static_cast<float>(lo);
            out[i + 16] = d * static_cast<float>(hi);
        }
    }
    return t;
}

Tensor Q41ToTensor(const QuanTensor& q) {
    if (q.type != QuantType::kQ41) {
        throw std::runtime_error("Q41ToTensor: expected Q4_1 QuantizedTensor");
    }
    if (q.shape.empty() && q.kQ41_data.empty()) {
        return Tensor();
    }
    const size_t num = q.GetNumSize();
    if (num % kQuantBlockSize != 0) {
        throw std::runtime_error("Q41ToTensor: element count not block-aligned");
    }
    const size_t n_blocks = num / kQuantBlockSize;
    if (q.kQ41_data.size() != n_blocks * kBytesPerQ41Block) {
        throw std::runtime_error("Q41ToTensor: data size mismatch");
    }
    Tensor t(q.shape);
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = q.kQ41_data.data() + b * kBytesPerQ41Block;
        const float    d   = Fp16ToFp32(ReadU16LE(blk));
        const float    m   = Fp16ToFp32(ReadU16LE(blk + 2));
        const uint8_t* qs  = blk + 4;
        float* out = t.data.data() + b * kQuantBlockSize;
        for (size_t i = 0; i < 16; ++i) {
            out[i]      = d * static_cast<float>(qs[i] & 0x0F) + m;
            out[i + 16] = d * static_cast<float>(qs[i] >> 4)   + m;
        }
    }
    return t;
}

}// namespace yan_lamma
