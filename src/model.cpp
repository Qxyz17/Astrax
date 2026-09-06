#include "astrax/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "astrax/architecture.hpp"
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

std::string ascii_lower(std::string value) {
    for (char& character : value) {
        if (character >= 'A' && character <= 'Z') {
            character = static_cast<char>(character - 'A' + 'a');
        }
    }
    return value;
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

osten::Config make_osten_config(const ModelConfig& config) {
    osten::Config result;
    result.state_dim = config.state_dim;
    result.goal_dim = config.goal_dim;
    result.memory_dim = config.state_dim;
    result.action_count = config.action_count;
    result.seed = config.seed;
    return result;
}

} // namespace

AstraxModel::AstraxModel(ModelConfig config)
    : config_(std::move(config)),
      state_{config_.version, 0, 0.0, 0, {}, {}},
      memory_(config_.memory_vector_dim, config_.memory_capacity),
      engine_(make_osten_config(config_)),
      predictor_(config_.state_dim, config_.action_count, config_.learning_rate),
      values_(config_.state_dim, config_.action_count, config_.learning_rate),
      intrinsic_(config_.intrinsic_reward_scale),
      trainer_(config_, predictor_, values_, intrinsic_),
      dialogue_() {}

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
    } else if (dialogue_.trained()) {
        output.text = dialogue_.respond(input.text, &output.confidence);
        if (output.text.empty()) {
            output.text = renderer_.render_text(input.text, output, state_);
        }
    } else {
        output.text = renderer_.render_text(input.text, output, state_);
    }
    state_.last_output = output.text;
    return output;
}

std::string AstraxModel::introspect(const std::string& question) const {
    const std::string normalized_question = ascii_lower(question);
    if (question.find("迭代") != std::string::npos ||
        normalized_question.find("iteration") != std::string::npos) {
        return std::to_string(state_.iteration);
    }
    if (question.find("目标") != std::string::npos ||
        normalized_question.find("goal") != std::string::npos) {
        return state_.current_goal.description;
    }
    if (question.find("记住") != std::string::npos ||
        normalized_question.find("memory") != std::string::npos ||
        normalized_question.find("memories") != std::string::npos ||
        normalized_question.find("remember") != std::string::npos) {
        if (question.find("什么") != std::string::npos ||
            normalized_question.find("what") != std::string::npos) {
            std::ostringstream remembered;
            for (const MemoryRecord& record : memory_.records()) {
                if (remembered.tellp() > 0) {
                    remembered << '\n';
                }
                remembered << record.content;
            }
            return remembered.str();
        }
        return std::to_string(memory_.size());
    }
    if (question.find("运行") != std::string::npos ||
        normalized_question.find("uptime") != std::string::npos) {
        return std::to_string(state_.uptime_seconds);
    }
    if (question.find("版本") != std::string::npos ||
        normalized_question.find("version") != std::string::npos) {
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

DialogueTrainingReport AstraxModel::train_dialogue(
    const std::vector<DialogueExample>& dataset, std::size_t epochs) {
    return dialogue_.train(dataset, epochs);
}

DialogueTrainingReport AstraxModel::train_dialogue_tsv(const std::string& path,
                                                       std::size_t epochs) {
    return train_dialogue(
        HolisticDialogueModel::load_tsv(path, dialogue_.max_response_bytes()),
        epochs);
}

std::string AstraxModel::chat(const std::string& input) {
    MultimodalInput message;
    message.modality = Modality::Text;
    message.text = input;
    return step(message, OutputMode::Text).text;
}

void AstraxModel::save_checkpoint(const std::string& path) const {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create Astrax checkpoint: " + path);
    }
    constexpr char magic[] = "ASTRAX-TRAINING-CHECKPOINT";
    constexpr std::uint32_t format_version = 2;
    output.write(magic, sizeof(magic));
    output.write(reinterpret_cast<const char*>(&format_version),
                 sizeof(format_version));
    const std::uint64_t state_dim = config_.state_dim;
    const std::uint64_t action_count = config_.action_count;
    output.write(reinterpret_cast<const char*>(&state_dim), sizeof(state_dim));
    output.write(reinterpret_cast<const char*>(&action_count), sizeof(action_count));
    predictor_.save(output);
    values_.save(output);
    intrinsic_.save(output);
    dialogue_.save(output);
    if (!output) {
        throw std::runtime_error("failed to write Astrax checkpoint: " + path);
    }
}

void AstraxModel::load_checkpoint(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open Astrax checkpoint: " + path);
    }
    constexpr char expected_magic[] = "ASTRAX-TRAINING-CHECKPOINT";
    char magic[sizeof(expected_magic)]{};
    std::uint32_t format_version = 0;
    std::uint64_t state_dim = 0;
    std::uint64_t action_count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    input.read(reinterpret_cast<char*>(&state_dim), sizeof(state_dim));
    input.read(reinterpret_cast<char*>(&action_count), sizeof(action_count));
    if (!input ||
        std::string(magic, sizeof(magic)) !=
            std::string(expected_magic, sizeof(expected_magic)) ||
        (format_version != 1 && format_version != 2) ||
        state_dim != config_.state_dim ||
        action_count != config_.action_count) {
        throw std::runtime_error("Astrax checkpoint header does not match model");
    }
    predictor_.load(input);
    values_.load(input);
    intrinsic_.load(input);
    if (format_version >= 2) {
        dialogue_.load(input);
    }
    if (!input) {
        throw std::runtime_error("Astrax checkpoint is incomplete: " + path);
    }
}

} // namespace astrax
