#include "astrax/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include "astrax/math.hpp"

namespace astrax {
namespace {

std::uint64_t hash_bytes(const std::string& value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string modality_name(Modality modality) {
    switch (modality) {
    case Modality::Text: return "text";
    case Modality::Code: return "code";
    case Modality::Image: return "image";
    case Modality::Audio: return "audio";
    case Modality::Video: return "video";
    case Modality::Binary: return "binary";
    }
    return "unknown";
}

} // namespace

AstraxModel::AstraxModel(ModelConfig config)
    : config_(std::move(config)),
      state_{config_.version, 0, 0.0, 0, {}, {}},
      memory_(config_.memory_vector_dim, config_.memory_capacity),
      engine_([&config_] {
          osten::Config result;
          result.state_dim = config.state_dim;
          result.goal_dim = config.goal_dim;
          result.memory_dim = config.state_dim;
          result.action_count = config.action_count;
          result.seed = config.seed;
          return result;
      }()),
      predictor_(config_.state_dim, config_.action_count, config_.learning_rate),
      values_(config_.state_dim, config_.action_count, config_.learning_rate),
      intrinsic_(config_.intrinsic_reward_scale),
      trainer_(config_, predictor_, values_, intrinsic_) {}

void AstraxModel::set_goal(Goal goal) {
    goal.priority = std::clamp(goal.priority, 0.0F, 1.0F);
    goal.active = true;
    state_.current_goal = std::move(goal);
}

void AstraxModel::clear_goal() {
    state_.current_goal = {};
}

math::Vector AstraxModel::embed(const MultimodalInput& input) const {
    if (!input.features.empty()) {
        math::Vector result(config_.memory_vector_dim, 0.0F);
        for (std::size_t index = 0; index < input.features.size(); ++index) {
            const std::size_t bucket = index % result.size();
            result[bucket] += input.features[index];
        }
        const float scale = 1.0F / std::sqrt(static_cast<float>(input.features.size()));
        for (float& value : result) {
            value = std::clamp(value * scale, -1.0F, 1.0F);
        }
        return result;
    }
    math::Vector result(config_.memory_vector_dim, 0.0F);
    const std::string value = modality_name(input.modality) + ":" + input.text +
                              ":" + input.metadata;
    for (std::size_t index = 0; index < value.size(); ++index) {
        const std::uint64_t hash = hash_bytes(value.substr(index, 1));
        result[hash % result.size()] += (hash & 1ULL) != 0ULL ? 1.0F : -1.0F;
    }
    const float norm = math::norm(result);
    if (norm > 1.0F) {
        for (float& item : result) {
            item /= norm;
        }
    }
    return result;
}

void AstraxModel::remember(const std::string& content, float salience) {
    MultimodalInput input;
    input.text = content;
    memory_.write(content, embed(input), salience, state_.iteration);
    state_.memory_count = memory_.size();
}

std::vector<MemoryRecord> AstraxModel::recall(const std::string& query, std::size_t limit) {
    MultimodalInput input;
    input.text = query;
    return memory_.retrieve(embed(input), limit, state_.iteration);
}

std::string AstraxModel::context_for(const math::Vector& query) {
    const std::vector<MemoryRecord> records = memory_.retrieve(
        query, 4, state_.iteration);
    std::ostringstream context;
    for (const MemoryRecord& record : records) {
        context << record.content << '\n';
    }
    return context.str();
}

ModelOutput AstraxModel::step(const MultimodalInput& input, OutputMode mode) {
    const auto started = std::chrono::steady_clock::now();
    const math::Vector query = embed(input);
    if (!input.text.empty() || !input.metadata.empty() || !input.features.empty()) {
        const std::string memory_text = input.text.empty()
            ? "[" + modality_name(input.modality) + "] " + input.metadata
            : input.text;
        memory_.write(memory_text, query, 0.6F, state_.iteration);
    }

    const std::string context = context_for(query);
    const std::string observation = input.text.empty()
        ? modality_name(input.modality) + " " + input.metadata + " " + context
        : input.text + "\n" + context;
    const osten::Decision decision = engine_.tick(observation);
    const float intrinsic_reward = intrinsic_.observe(decision.state);

    ++state_.iteration;
    state_.memory_count = memory_.size();
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started);
    state_.uptime_seconds += elapsed.count();

    ModelOutput output;
    output.iteration = state_.iteration;
    output.mode = mode;
    output.action_id = decision.action_id;
    output.confidence = decision.confidence;
    output.intrinsic_reward = intrinsic_reward;
    output.state = decision.state;
    output.goal = decision.goal;
    if (mode == OutputMode::Code) {
        output.text = renderer_.render_code(input.text, output, state_);
    } else if (mode == OutputMode::State) {
        output.text = renderer_.render_state(state_);
    } else {
        output.text = renderer_.render_text(input.text, output, state_);
    }
    state_.last_output = output.text;
    return output;
}

std::string AstraxModel::introspect(const std::string& question) const {
    if (question.find("迭代") != std::string::npos ||
        question.find("iteration") != std::string::npos) {
        return std::to_string(state_.iteration);
    }
    if (question.find("目标") != std::string::npos ||
        question.find("goal") != std::string::npos) {
        return state_.current_goal.description;
    }
    if (question.find("记住") != std::string::npos ||
        question.find("memory") != std::string::npos) {
        return std::to_string(memory_.size());
    }
    if (question.find("运行") != std::string::npos ||
        question.find("uptime") != std::string::npos) {
        return std::to_string(state_.uptime_seconds);
    }
    if (question.find("版本") != std::string::npos ||
        question.find("version") != std::string::npos) {
        return state_.version;
    }
    return "No structured introspection field matched.";
}

TrainingReport AstraxModel::train_offline(const std::vector<OfflineTransition>& dataset,
                                          std::size_t epochs) {
    return trainer_.train(dataset, epochs);
}

TrainingReport AstraxModel::train_offline_csv(const std::string& path,
                                              std::size_t epochs) {
    return train_offline(
        OfflineRLTrainer::load_csv(path, config_.state_dim), epochs);
}

} // namespace astrax
