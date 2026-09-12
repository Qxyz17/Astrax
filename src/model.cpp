#include "astrax/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <unordered_set>

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

float centered_scale(const math::Vector& values) {
    if (values.empty()) {
        return 1.0F;
    }
    const float mean = std::accumulate(values.begin(), values.end(), 0.0F) /
                       static_cast<float>(values.size());
    float scale = 0.0F;
    for (const float value : values) {
        scale = std::max(scale, std::abs(value - mean));
    }
    return std::max(scale, 1.0e-6F);
}

float centered_value(float value, const math::Vector& values, float scale) {
    const float mean = std::accumulate(values.begin(), values.end(), 0.0F) /
                       static_cast<float>(values.size());
    return (value - mean) / scale;
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

std::vector<TextDocument> bounded_dialogue_dataset(
    const std::vector<TextDocument>& dataset, std::size_t limit) {
    if (limit == 0U || dataset.size() <= limit) {
        return dataset;
    }
    std::vector<TextDocument> unique;
    unique.reserve(std::min(dataset.size(), limit));
    std::unordered_set<std::string> seen;
    for (const TextDocument& document : dataset) {
        if (seen.insert(document.text).second) {
            unique.push_back(document);
        }
    }
    if (unique.size() <= limit) {
        return unique;
    }
    std::vector<TextDocument> sampled;
    sampled.reserve(limit);
    for (std::size_t index = 0; index < limit; ++index) {
        const std::size_t source = index * unique.size() / limit;
        sampled.push_back(unique[source]);
    }
    return sampled;
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
      state_{0, 0.0, 0, {}, {}},
      memory_(config_.memory_vector_dim, config_.memory_capacity),
      engine_(make_osten_config(config_)),
      predictor_(config_.state_dim, config_.action_count, config_.learning_rate),
      values_(config_.state_dim, config_.action_count, config_.learning_rate),
      intrinsic_(config_.intrinsic_reward_scale),
      trainer_(config_, predictor_, values_, intrinsic_),
      // This is still a compact parallel decoder, not a token generator.
      // More input features and slots are needed for mixed Chinese/English
      // text and short code blocks without falling back to a fixed renderer.
      dialogue_(512, 256, std::max(config_.learning_rate, 0.08F), 128) {
    if (config_.state_dim == 0 || config_.goal_dim == 0 ||
        config_.action_count == 0 || config_.memory_vector_dim == 0 ||
        config_.memory_capacity == 0 || !std::isfinite(config_.discount) ||
        config_.discount < 0.0F || config_.discount > 1.0F ||
        !std::isfinite(config_.learning_rate) || config_.learning_rate <= 0.0F ||
        !std::isfinite(config_.policy_guidance_weight) ||
        !std::isfinite(config_.value_guidance_weight)) {
        throw std::invalid_argument("invalid Astrax model configuration");
    }
}

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
            result[index % result.size()] += input.features[index];
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

std::size_t AstraxModel::select_action(const osten::Decision& decision) const {
    if (decision.action_logits.empty()) {
        throw std::invalid_argument("Osten decision has no action logits");
    }
    if (decision.state.size() != config_.state_dim) {
        throw std::invalid_argument("Osten decision state dimension does not match model");
    }
    const math::Vector learned_values = values_.values(decision.state);
    if (learned_values.size() != decision.action_logits.size()) {
        throw std::invalid_argument("Osten and Astrax action dimensions do not match");
    }

    // Normalize each stream per decision. Osten logits and offline Q values do
    // not share a calibrated numeric scale; centering makes the combination
    // meaningful without replacing Osten or using a hand-written action rule.
    const float policy_scale = centered_scale(decision.action_logits);
    const float value_scale = centered_scale(learned_values);
    std::size_t selected = decision.action_id < decision.action_logits.size()
        ? decision.action_id : 0U;
    float best_score = -std::numeric_limits<float>::infinity();
    for (std::size_t action = 0; action < decision.action_logits.size(); ++action) {
        const float score = config_.policy_guidance_weight *
                centered_value(decision.action_logits[action],
                               decision.action_logits, policy_scale) +
            config_.value_guidance_weight *
                centered_value(learned_values[action], learned_values, value_scale);
        if (score > best_score ||
            (score == best_score && action == decision.action_id)) {
            best_score = score;
            selected = action;
        }
    }
    return selected;
}

ModelOutput AstraxModel::step(const MultimodalInput& input, OutputMode mode) {
    const auto started = std::chrono::steady_clock::now();
    const math::Vector query = embed(input);
    // Retrieve evidence before writing the current observation. An input must
    // not be able to cite itself as memory during the same decision.
    const std::string context = context_for(query);
    const std::string observation = input.text.empty()
        ? modality_name(input.modality) + " " + input.metadata + " " + context
        : input.text + "\n" + context;
    const osten::Decision decision = engine_.tick(observation);
    const std::size_t selected_action = select_action(decision);
    const float intrinsic_reward = intrinsic_.observe(decision.state);

    ++state_.iteration;
    state_.memory_count = memory_.size();
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started);
    state_.uptime_seconds += elapsed.count();

    ModelOutput output;
    output.iteration = state_.iteration;
    output.mode = mode;
    output.action_id = selected_action;
    output.engine_action_id = decision.action_id;
    output.action_value = values_.value(decision.state, selected_action);
    output.action_values = values_.values(decision.state);
    output.confidence = decision.confidence;
    output.intrinsic_reward = intrinsic_reward;
    output.state = decision.state;
    output.goal = decision.goal;
    if (mode == OutputMode::State) {
        output.text = renderer_.render_state(state_);
    } else if (dialogue_.trained()) {
        output.text = dialogue_.respond(
            input.text, decision.state, selected_action, output.action_value,
            decision.goal, context, &output.confidence);
    } else {
        output.text = renderer_.render_decision(output, state_);
    }
    state_.last_output = output.text;
    if (!input.text.empty() || !input.metadata.empty() || !input.features.empty()) {
        const std::string memory_text = input.text.empty()
            ? "[" + modality_name(input.modality) + "] " + input.metadata
            : input.text;
        memory_.write(memory_text, query, 0.6F, state_.iteration);
    }
    pending_state_ = decision.state;
    pending_action_ = selected_action;
    has_pending_transition_ = true;
    return output;
}

float AstraxModel::observe_feedback(float reward, bool terminal) {
    return observe_feedback(pending_state_, reward, terminal);
}

float AstraxModel::observe_feedback(const math::Vector& next_state,
                                    float reward, bool terminal) {
    if (!has_pending_transition_) {
        throw std::logic_error("no pending transition to observe");
    }
    if (!std::isfinite(reward)) {
        throw std::invalid_argument("feedback reward must be finite");
    }
    math::require_size(next_state, config_.state_dim, "feedback next_state");
    for (const float value : next_state) {
        if (!std::isfinite(value) || value < -1.0F || value > 1.0F) {
            throw std::invalid_argument("feedback next_state must be finite in [-1, 1]");
        }
    }
    OfflineTransition transition;
    transition.state = pending_state_;
    transition.action = pending_action_;
    transition.next_state = next_state;
    transition.reward = reward;
    transition.terminal = terminal;
    const float loss = values_.train_q_learning(transition, config_.discount,
                                                transition.next_state);
    has_pending_transition_ = false;
    return loss;
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
        return {};
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
    const std::vector<TextDocument>& dataset, std::size_t epochs) {
    const std::vector<TextDocument> bounded = bounded_dialogue_dataset(
        dataset, config_.dialogue_training_document_limit);
    return dialogue_.train(bounded, epochs);
}

DialogueTrainingReport AstraxModel::train_dialogue_documents(
    const std::string& path, std::size_t epochs) {
    return train_dialogue(
        HolisticDialogueModel::load_documents(path, 4U * 1024U * 1024U), epochs);
}

DialogueTrainingReport AstraxModel::train_dialogue_pairs(
    const std::string& path, std::size_t epochs) {
    return dialogue_.train_pairs(
        HolisticDialogueModel::load_pairs(path, 4U * 1024U * 1024U), epochs);
}

DialogueTrainingReport AstraxModel::train_dialogue_pairs(
    const std::vector<TextDocument>& dataset, std::size_t epochs) {
    const std::size_t limit = config_.dialogue_training_document_limit;
    if (limit != 0U && dataset.size() > limit) {
        std::vector<TextDocument> bounded(dataset.begin(),
                                          dataset.begin() + static_cast<std::ptrdiff_t>(limit));
        return dialogue_.train_pairs(bounded, epochs);
    }
    return dialogue_.train_pairs(dataset, epochs);
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
    constexpr std::uint32_t format_version = 4;
    output.write(magic, sizeof(magic));
    output.write(reinterpret_cast<const char*>(&format_version), sizeof(format_version));
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
    if (!input || std::string(magic, sizeof(magic)) !=
            std::string(expected_magic, sizeof(expected_magic)) ||
        format_version != 4 || state_dim != config_.state_dim ||
        action_count != config_.action_count) {
        throw std::runtime_error("Astrax checkpoint header does not match model");
    }
    predictor_.load(input);
    values_.load(input);
    intrinsic_.load(input);
    dialogue_.load(input);
    if (!input) {
        throw std::runtime_error("Astrax checkpoint is incomplete: " + path);
    }
}

void AstraxModel::load_checkpoint_bytes(const std::vector<std::uint8_t>& bytes) {
    std::string payload(bytes.begin(), bytes.end());
    std::istringstream input(payload, std::ios::binary);
    constexpr char expected_magic[] = "ASTRAX-TRAINING-CHECKPOINT";
    char magic[sizeof(expected_magic)]{};
    std::uint32_t format_version = 0;
    std::uint64_t state_dim = 0;
    std::uint64_t action_count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    input.read(reinterpret_cast<char*>(&state_dim), sizeof(state_dim));
    input.read(reinterpret_cast<char*>(&action_count), sizeof(action_count));
    if (!input || std::string(magic, sizeof(magic)) !=
            std::string(expected_magic, sizeof(expected_magic)) ||
        format_version != 4 || state_dim != config_.state_dim ||
        action_count != config_.action_count) {
        throw std::runtime_error("embedded Astrax checkpoint header does not match model");
    }
    predictor_.load(input);
    values_.load(input);
    intrinsic_.load(input);
    dialogue_.load(input);
    if (!input) {
        throw std::runtime_error("embedded Astrax checkpoint is incomplete");
    }
}

} // namespace astrax
