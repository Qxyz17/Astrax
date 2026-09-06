#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "astrax/math.hpp"

namespace astrax {

enum class Modality {
    Text,
    Code,
    Image,
    Audio,
    Video,
    Binary
};

enum class OutputMode {
    Text,
    Code,
    State
};

struct ModelConfig {
    std::string version = "0.1.1-alpha";
    std::size_t state_dim = 32;
    std::size_t goal_dim = 16;
    std::size_t action_count = 8;
    std::size_t memory_vector_dim = 64;
    std::size_t memory_capacity = 256;
    float discount = 0.97F;
    float learning_rate = 0.02F;
    float intrinsic_reward_scale = 0.20F;
    std::uint64_t seed = 23;
};

struct Goal {
    std::string id;
    std::string description;
    float priority = 0.5F;
    bool active = true;
};

struct AstraxState {
    std::string version;
    std::uint64_t iteration = 0;
    double uptime_seconds = 0.0;
    std::size_t memory_count = 0;
    Goal current_goal;
    std::string last_output;
};

struct MemoryRecord {
    std::uint64_t id = 0;
    std::string content;
    math::Vector embedding;
    float salience = 0.5F;
    std::uint64_t created_at = 0;
    std::uint64_t last_used = 0;
    std::size_t use_count = 0;
};

struct MultimodalInput {
    Modality modality = Modality::Text;
    std::string text;
    math::Vector features;
    std::string metadata;
};

struct ModelOutput {
    std::uint64_t iteration = 0;
    OutputMode mode = OutputMode::Text;
    std::string text;
    std::size_t action_id = 0;
    float confidence = 0.0F;
    float intrinsic_reward = 0.0F;
    math::Vector state;
    math::Vector goal;
};

struct OfflineTransition {
    math::Vector state;
    std::size_t action = 0;
    float reward = 0.0F;
    math::Vector next_state;
    bool terminal = false;
};

struct TrainingReport {
    std::size_t samples = 0;
    std::size_t epochs = 0;
    float predictor_loss = 0.0F;
    float value_loss = 0.0F;
    float average_intrinsic_reward = 0.0F;
};

} // namespace astrax

