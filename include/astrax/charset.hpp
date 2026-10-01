#pragma once

#include <cstddef>
#include <cstdint>

namespace astrax::charset {

// Astrax emits UTF-8 bytes. A response slot selects one of 256 byte values or
// the reserved stop class, so the output layer is a small dense distribution
// (257 classes) instead of a factorized 21-bit codepoint space. This is still
// a parallel whole-sequence denoiser: every slot is refined in parallel and no
// slot performs left-to-right next-token generation. Byte units simply make
// the learning problem tractable on a CPU-sized model.

inline constexpr std::size_t kByteCount = 256U;
// Class 0..255 are byte values; class 256 is the stop/blank sentinel.
inline constexpr std::size_t kClassCount = kByteCount + 1U;
inline constexpr std::size_t kBlankClass = kByteCount;

inline constexpr bool is_byte_class(std::size_t class_index) {
    return class_index < kByteCount;
}

} // namespace astrax::charset
