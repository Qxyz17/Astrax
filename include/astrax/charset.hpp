#pragma once

#include <cstddef>
#include <cstdint>

namespace astrax::charset {

// Astrax tells the model the complete UTF-8/Unicode codepoint space and lets
// the model select any codepoint itself. There is no data-derived vocabulary
// and no truncation to a "common characters" table: the full range is
// available, and the output layer is factorized so that a full 21-bit
// codepoint can be predicted without a 1.1M-class dense softmax.
//
// Factorization: a codepoint is split into a high part and a low part.
//   high = codepoint >> low_bits
//   low  = codepoint & (low_count - 1)
// The model predicts a high distribution and a low distribution per response
// slot, and the pair is combined back into a codepoint. This keeps the
// non-Transformer, non-next-token, parallel whole-sequence contract intact.

inline constexpr std::uint32_t kMaxCodepoint = 0x10FFFFU;
inline constexpr std::size_t kCodepointBits = 21U;
// The split point trades the high-class count against the low-class count.
// 10 low bits keep the dominant low head small (1024 classes) while the high
// head stays at 1088 classes, so the factorized output layer is about 400K
// parameters instead of several million. The full 21-bit Unicode range is
// still covered; only the split ratio changes.
inline constexpr std::size_t kLowBits = 10U;
inline constexpr std::uint32_t kLowCount = 1U << kLowBits;      // 1024
inline constexpr std::uint32_t kLowMask = kLowCount - 1U;
inline constexpr std::uint32_t kHighCount = (kMaxCodepoint >> kLowBits) + 1U; // 1088

inline constexpr bool is_valid_codepoint(std::uint32_t codepoint) {
    return codepoint <= kMaxCodepoint &&
           !(codepoint >= 0xD800U && codepoint <= 0xDFFFU);
}

inline constexpr std::uint32_t high_of(std::uint32_t codepoint) {
    return codepoint >> kLowBits;
}

inline constexpr std::uint32_t low_of(std::uint32_t codepoint) {
    return codepoint & kLowMask;
}

inline constexpr std::uint32_t combine(std::uint32_t high, std::uint32_t low) {
    return (high << kLowBits) | (low & kLowMask);
}

// The reserved "no output / stop" class. A response slot that selects this
// class ends the response at that position.
inline constexpr std::uint32_t kBlankClass = 0U;

} // namespace astrax::charset
