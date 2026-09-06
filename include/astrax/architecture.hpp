#pragma once

namespace astrax::architecture {

// Repository-level hard constraints. Keep these literals explicit so tests and
// downstream tools can verify that the intended architecture has not drifted.
inline constexpr bool kUsesTransformer = false;
inline constexpr bool kUsesNextTokenPrediction = false;
inline constexpr bool kUsesAutoregressiveGeneration = false;
inline constexpr bool kOstenIsCoreDecisionEngine = true;
inline constexpr bool kFinalDecisionUsesKeywordRules = false;
inline constexpr bool kFinalDecisionUsesDirectRetrieval = false;
inline constexpr bool kUsesParallelResponseSlots = true;

static_assert(!kUsesTransformer, "Astrax must remain non-Transformer");
static_assert(!kUsesNextTokenPrediction,
              "Astrax must not use next-token prediction");
static_assert(!kUsesAutoregressiveGeneration,
              "Astrax must not use autoregressive generation");
static_assert(kOstenIsCoreDecisionEngine,
              "Osten must remain the core decision engine");
static_assert(!kFinalDecisionUsesKeywordRules,
              "keywords must not directly select the final answer");
static_assert(!kFinalDecisionUsesDirectRetrieval,
              "retrieval must not directly select the final answer");
static_assert(kUsesParallelResponseSlots,
              "dialogue must decode complete response slots in parallel");

} // namespace astrax::architecture
