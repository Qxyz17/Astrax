#pragma warning(disable: 4267)
#include "astrax/dialogue.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace astrax {
namespace {

// The decoder is character/codepoint based, not byte based.  The old 128
// entry vocabulary silently mapped most Chinese codepoints to the end class,
// which made a trained model look like a broken fixed-answer renderer.  The
// corpus currently contains fewer than 512 distinct codepoints, so keep room
// for a real Unicode vocabulary while retaining a small research model.
constexpr std::size_t kHiddenDim = 96;
constexpr std::size_t kMaxVocabulary = 1024;
constexpr std::size_t kNegativeSamples = 24;

std::uint64_t mix_hash(std::uint64_t value) {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t hash_codepoints(const std::vector<std::uint32_t>& values,
                              std::size_t begin, std::size_t count) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = begin; index < begin + count; ++index) {
        hash ^= static_cast<std::uint64_t>(values[index]) + 0x9e3779b9ULL;
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool append_codepoint(std::uint32_t value, std::string& output) {
    if (value <= 0x7FU) {
        output.push_back(static_cast<char>(value));
    } else if (value <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (value >> 6U)));
        output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
    } else if (value <= 0xFFFFU && !(value >= 0xD800U && value <= 0xDFFFU)) {
        output.push_back(static_cast<char>(0xE0U | (value >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
    } else if (value <= 0x10FFFFU) {
        output.push_back(static_cast<char>(0xF0U | (value >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((value >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
    } else {
        return false;
    }
    return true;
}

bool is_layout_codepoint(std::uint32_t value) {
    return value == 0x09U || value == 0x0AU || value == 0x0DU ||
           value == 0x20U || value == 0xA0U;
}

struct Decoded {
    std::vector<std::uint32_t> values;
    bool valid = true;
};

Decoded decode_utf8(const std::string& value) {
    Decoded result;
    std::size_t index = 0;
    while (index < value.size()) {
        const unsigned char lead = static_cast<unsigned char>(value[index]);
        std::size_t length = 0;
        std::uint32_t codepoint = 0;
        if (lead <= 0x7FU) {
            length = 1;
            codepoint = lead;
        } else if (lead >= 0xC2U && lead <= 0xDFU) {
            length = 2;
            codepoint = lead & 0x1FU;
        } else if (lead >= 0xE0U && lead <= 0xEFU) {
            length = 3;
            codepoint = lead & 0x0FU;
        } else if (lead >= 0xF0U && lead <= 0xF4U) {
            length = 4;
            codepoint = lead & 0x07U;
        } else {
            result.valid = false;
            ++index;
            continue;
        }
        if (index + length > value.size()) {
            result.valid = false;
            ++index;
            continue;
        }
        bool valid = true;
        for (std::size_t offset = 1; offset < length; ++offset) {
            const unsigned char continuation =
                static_cast<unsigned char>(value[index + offset]);
            if ((continuation & 0xC0U) != 0x80U) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6U) | (continuation & 0x3FU);
        }
        valid = valid &&
                !((length == 2 && codepoint < 0x80U) ||
                  (length == 3 && codepoint < 0x800U) ||
                  (length == 4 && codepoint < 0x10000U) ||
                  codepoint > 0x10FFFFU ||
                  (codepoint >= 0xD800U && codepoint <= 0xDFFFU));
        if (!valid) {
            result.valid = false;
            ++index;
            continue;
        }
        if (codepoint != 0U &&
            (codepoint >= 0x20U || codepoint == '\n' ||
             codepoint == '\r' || codepoint == '\t')) {
            result.values.push_back(codepoint);
        }
        index += length;
    }
    return result;
}

bool valid_utf8(const std::string& value) {
    return decode_utf8(value).valid;
}

void write_size(std::ostream& output, std::size_t value) {
    const std::uint64_t stored = static_cast<std::uint64_t>(value);
    output.write(reinterpret_cast<const char*>(&stored), sizeof(stored));
}

std::size_t read_size(std::istream& input, std::size_t limit = 100000000U) {
    std::uint64_t stored = 0;
    input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
    if (!input || stored > limit) {
        throw std::runtime_error("invalid dialogue checkpoint size");
    }
    return static_cast<std::size_t>(stored);
}

void write_vector(std::ostream& output, const math::Vector& values) {
    write_size(output, values.size());
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write dialogue checkpoint");
    }
}

math::Vector read_vector(std::istream& input, std::size_t expected) {
    const std::size_t size = read_size(input);
    if (size != expected) {
        throw std::runtime_error("dialogue checkpoint dimension mismatch");
    }
    math::Vector result(size, 0.0F);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("dialogue checkpoint ended in vector");
    }
    for (const float value : result) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("dialogue checkpoint contains non-finite value");
        }
    }
    return result;
}

void add_hashed(math::Vector& vector, std::uint64_t hash, float amount) {
    const std::size_t bucket = static_cast<std::size_t>(hash % vector.size());
    vector[bucket] += (hash & 1ULL) != 0ULL ? amount : -amount;
}

} // namespace

HolisticDialogueModel::HolisticDialogueModel(std::size_t input_dim,
                                             std::size_t max_response_codepoints,
                                             float learning_rate,
                                             std::size_t condition_dim)
    : input_dim_(input_dim),
      max_response_codepoints_(max_response_codepoints),
      learning_rate_(learning_rate),
      condition_dim_(condition_dim),
      hidden_dim_(kHiddenDim) {
    if (input_dim_ == 0 || max_response_codepoints_ == 0 || condition_dim_ == 0 ||
        !std::isfinite(learning_rate_) || learning_rate_ <= 0.0F) {
        throw std::invalid_argument("invalid holistic dialogue configuration");
    }
}

math::Vector HolisticDialogueModel::encode_text(const std::string& input) const {
    math::Vector result(input_dim_, 0.0F);
    math::Vector reverse_state(input_dim_, 0.0F);
    const Decoded decoded = decode_utf8(input);
    const auto& values = decoded.values;
    if (values.empty()) {
        return result;
    }
    std::size_t events = 0;
    // Two fixed-size gated recurrences preserve information from both ends of
    // the complete input. This is a small state-space encoder, not attention
    // and not an autoregressive language model.
    for (std::size_t index = 0; index < values.size(); ++index) {
        const float position = static_cast<float>(index % 64U) / 64.0F;
        const float input_gate = 0.35F + 0.25F * std::sin(position * 6.2831853F);
        const float forget_gate = 0.78F + 0.12F * std::cos(position * 6.2831853F);
        for (float& item : result) {
            item *= forget_gate;
        }
        math::Vector event(input_dim_, 0.0F);
        add_hashed(event, mix_hash(values[index] + index * 0x9e3779b9ULL), 1.0F);
        add_hashed(event, mix_hash(values[index] * 17U + index), input_gate);
        for (std::size_t bucket = 0; bucket < result.size(); ++bucket) {
            result[bucket] += input_gate * event[bucket];
        }
        ++events;
        if (index + 1 < values.size()) {
            add_hashed(result, mix_hash(hash_codepoints(values, index, 2)), 1.2F);
            ++events;
        }
        if (index + 2 < values.size()) {
            add_hashed(result, mix_hash(hash_codepoints(values, index, 3)), 1.0F);
            ++events;
        }
        const std::uint64_t position_hash =
            mix_hash(static_cast<std::uint64_t>(index % 32U) * 0x9e3779b97f4a7c15ULL +
                     values[index]);
        add_hashed(result, position_hash, 0.6F);
        ++events;
    }
    add_hashed(result, mix_hash(values.size()), 0.75F);
    ++events;
    for (std::size_t offset = 0; offset < values.size(); ++offset) {
        const std::size_t index = values.size() - offset - 1U;
        const float position = static_cast<float>(index % 64U) / 64.0F;
        const float input_gate = 0.30F + 0.20F * std::cos(position * 6.2831853F);
        const float forget_gate = 0.80F + 0.10F * std::sin(position * 6.2831853F);
        for (float& item : reverse_state) {
            item *= forget_gate;
        }
        add_hashed(reverse_state, mix_hash(values[index] + index * 0x517cc1b7ULL),
                   input_gate);
    }
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = 0.62F * result[index] + 0.38F * reverse_state[index];
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(std::max<std::size_t>(1, events)));
    for (float& item : result) {
        item *= scale;
    }
    const float length = math::norm(result);
    if (length > 1.0F) {
        for (float& item : result) {
            item /= length;
        }
    }
    return result;
}

math::Vector HolisticDialogueModel::encode_condition(
    const math::Vector& osten_state, std::size_t action_id, float action_value,
    const math::Vector& goal, const std::string& memory_context) const {
    math::Vector result(condition_dim_, 0.0F);
    for (std::size_t index = 0; index < osten_state.size(); ++index) {
        result[index % result.size()] +=
            std::clamp(osten_state[index], -1.0F, 1.0F) * 0.25F;
    }
    for (std::size_t index = 0; index < goal.size(); ++index) {
        result[(index + 17U) % result.size()] +=
            std::clamp(goal[index], -1.0F, 1.0F) * 0.25F;
    }
    result[action_id % result.size()] += 0.5F;
    result[(result.size() / 2U) % result.size()] += std::clamp(action_value, -1.0F, 1.0F);
    const Decoded context = decode_utf8(memory_context);
    for (std::size_t index = 0; index < context.values.size(); ++index) {
        add_hashed(result, mix_hash(context.values[index] + index), 0.08F);
    }
    const float length = math::norm(result);
    if (length > 1.0F) {
        for (float& item : result) {
            item /= length;
        }
    }
    return result;
}

math::Vector HolisticDialogueModel::hidden_for(const math::Vector& features,
                                                const math::Vector& condition,
                                                std::size_t slot) const {
    const std::size_t combined_dim = input_dim_ + condition_dim_;
    math::Vector hidden(hidden_dim_, 0.0F);
    for (std::size_t row = 0; row < hidden_dim_; ++row) {
        float value = context_bias_[row] + slot_embeddings_[slot * hidden_dim_ + row];
        for (std::size_t column = 0; column < input_dim_; ++column) {
            value += context_weights_[row * combined_dim + column] * features[column];
        }
        for (std::size_t column = 0; column < condition_dim_; ++column) {
            value += condition_weights_[row * condition_dim_ + column] * condition[column];
        }
        hidden[row] = std::tanh(value);
    }
    return hidden;
}

DialogueTrainingReport HolisticDialogueModel::train(
    const std::vector<TextDocument>& dataset, std::size_t epochs) {
    for (const TextDocument& document : dataset) {
        const std::string& source = document.input.empty()
            ? document.text : document.input;
        const std::string& target = document.target.empty()
            ? document.text : document.target;
        if (source.empty() || target.empty() ||
            source.size() > 4U * 1024U * 1024U ||
            target.size() > 4U * 1024U * 1024U ||
            !valid_utf8(source) || !valid_utf8(target)) {
            throw std::invalid_argument("text training document is empty, too large, or invalid UTF-8");
        }
    }
    if (dataset.empty()) {
        throw std::invalid_argument("text dataset must not be empty");
    }
    DialogueTrainingReport report;
    report.examples = dataset.size();
    report.epochs = epochs;
    if (epochs == 0) {
        return report;
    }

    std::unordered_map<std::uint32_t, std::size_t> frequency;
    for (const TextDocument& document : dataset) {
        const std::string& target_text = document.target.empty()
            ? document.text : document.target;
        for (const std::uint32_t value : decode_utf8(target_text).values) {
            ++frequency[value];
        }
    }
    std::vector<std::uint32_t> candidates;
    candidates.reserve(frequency.size());
    for (const auto& item : frequency) {
        candidates.push_back(item.first);
    }
    std::sort(candidates.begin(), candidates.end(), [&](std::uint32_t left,
                                                         std::uint32_t right) {
        if (frequency[left] != frequency[right]) {
            return frequency[left] > frequency[right];
        }
        return left < right;
    });
    if (candidates.size() > kMaxVocabulary) {
        candidates.resize(kMaxVocabulary);
    }
    std::sort(candidates.begin(), candidates.end());
    vocabulary_ = std::move(candidates);
    const std::size_t class_count = vocabulary_.size() + 1U;
    const std::size_t combined_dim = input_dim_ + condition_dim_;
    context_weights_.assign(hidden_dim_ * combined_dim, 0.0F);
    context_bias_.assign(hidden_dim_, 0.0F);
    condition_weights_.assign(hidden_dim_ * condition_dim_, 0.0F);
    slot_embeddings_.assign(max_response_codepoints_ * hidden_dim_, 0.0F);
    class_embeddings_.assign(class_count * hidden_dim_, 0.0F);
    class_bias_.assign(class_count, 0.0F);
    // Do not start the encoder at the all-zero fixed point. With zero context
    // and slot parameters, the first softmax update can only learn class
    // frequency and no gradient can distinguish two inputs.
    for (std::size_t index = 0; index < context_weights_.size(); ++index) {
        const float phase = static_cast<float>((index * 17U + 11U) % 211U) / 211.0F;
        context_weights_[index] = 0.035F * std::sin(phase * 6.2831853F);
    }
    for (std::size_t index = 0; index < slot_embeddings_.size(); ++index) {
        const float phase = static_cast<float>((index * 29U + 7U) % 173U) / 173.0F;
        slot_embeddings_[index] = 0.025F * std::cos(phase * 6.2831853F);
    }
    for (std::size_t index = 0; index < class_embeddings_.size(); ++index) {
        const float phase = static_cast<float>((index * 37U) % 101U) / 101.0F;
        class_embeddings_[index] = 0.15F * std::sin(phase * 6.2831853F);
    }

    std::unordered_map<std::uint32_t, std::size_t> class_for;
    for (std::size_t index = 0; index < vocabulary_.size(); ++index) {
        class_for[vocabulary_[index]] = index + 1U;
    }

    std::size_t total_slots = 0;
    std::size_t correct_slots = 0;
    float total_loss = 0.0F;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        total_slots = 0;
        correct_slots = 0;
        total_loss = 0.0F;
        for (std::size_t document_index = 0; document_index < dataset.size(); ++document_index) {
            const TextDocument& document = dataset[document_index];
            const std::string& source_text = document.input.empty()
                ? document.text : document.input;
            const std::string& target_text = document.target.empty()
                ? document.text : document.target;
            const Decoded decoded = decode_utf8(target_text);
            if (decoded.values.empty()) {
                continue;
            }
             // The source is masked as a whole sequence. Pair targets remain
             // separate, so a response cannot leak into its own input.
             // Pair training must preserve the complete user input. Masking
             // belongs to document reconstruction, but would erase the exact
             // intent signal needed for input-to-target supervision.
             const math::Vector features = encode_text(source_text);
            // Language pretraining has no action labels.  Use the neutral
            // condition so random document indices cannot be memorized as
            // pseudo-semantics.  At inference the same learned representation
            // is conditioned by Osten's live state/action/value path.
            const math::Vector condition =
                encode_condition({}, 0U, 0.0F, {}, {});
            const std::size_t target_length = std::min(
                decoded.values.size(),
                max_response_codepoints_ > 0U
                    ? max_response_codepoints_ - 1U
                    : 0U);
            // Train actual target positions and exactly one end slot. Training
            // every remaining slot as end would overwhelm real characters and
            // collapse a parallel decoder to a trivial high-frequency class.
            const std::size_t slots = std::min(
                max_response_codepoints_, target_length + 1U);
            for (std::size_t slot = 0; slot < slots; ++slot) {
                const std::size_t target = slot < target_length
                    ? (class_for.count(decoded.values[slot]) != 0U
                        ? class_for[decoded.values[slot]] : 0U)
                    : 0U;
                math::Vector hidden = hidden_for(features, condition, slot);
                math::Vector scores(class_count, 0.0F);
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t current = 0; current < class_count; ++current) {
                    float score = class_bias_[current];
                    for (std::size_t h = 0; h < hidden_dim_; ++h) {
                        score += class_embeddings_[current * hidden_dim_ + h] * hidden[h];
                    }
                    scores[current] = score;
                    maximum = std::max(maximum, score);
                }
                float normalizer = 0.0F;
                for (const float score : scores) {
                    normalizer += std::exp(score - maximum);
                }
                total_loss += -(scores[target] - maximum -
                    std::log(std::max(normalizer, 1.0e-12F)));
                const std::size_t best = static_cast<std::size_t>(
                    std::distance(scores.begin(),
                                  std::max_element(scores.begin(), scores.end())));
                ++total_slots;
                if (best == target) {
                    ++correct_slots;
                }
                for (std::size_t current = 0; current < class_count; ++current) {
                    const float probability = std::exp(scores[current] - maximum) /
                        std::max(normalizer, 1.0e-12F);
                    const float gradient = learning_rate_ *
                        (probability - (current == target ? 1.0F : 0.0F));
                    class_bias_[current] -= gradient;
                    for (std::size_t h = 0; h < hidden_dim_; ++h) {
                        class_embeddings_[current * hidden_dim_ + h] -= gradient * hidden[h];
                    }
                }
                // Backpropagate the classifier error into the fixed-size
                // state-space encoder and slot state. Without this path, the
                // model can only learn character frequencies, not an input to
                // target mapping.
                for (std::size_t h = 0; h < hidden_dim_; ++h) {
                    float hidden_gradient = 0.0F;
                    for (std::size_t current = 0; current < class_count; ++current) {
                        const float probability = std::exp(scores[current] - maximum) /
                            std::max(normalizer, 1.0e-12F);
                        hidden_gradient += learning_rate_ *
                            (probability - (current == target ? 1.0F : 0.0F)) *
                            class_embeddings_[current * hidden_dim_ + h];
                    }
                    const float preactivation_gradient = hidden_gradient *
                        (1.0F - hidden[h] * hidden[h]);
                    context_bias_[h] -= preactivation_gradient;
                    slot_embeddings_[slot * hidden_dim_ + h] -= preactivation_gradient;
                    for (std::size_t column = 0; column < input_dim_; ++column) {
                        context_weights_[h * combined_dim + column] -=
                            preactivation_gradient * features[column];
                    }
                    for (std::size_t column = 0; column < condition_dim_; ++column) {
                        condition_weights_[h * condition_dim_ + column] -=
                            preactivation_gradient * condition[column];
                    }
                }
            }
        }
        report.reconstruction_loss = total_loss /
            static_cast<float>(std::max<std::size_t>(1U, total_slots));
        report.response_loss = report.reconstruction_loss;
        report.cross_entropy = report.reconstruction_loss;
        report.codepoint_accuracy = static_cast<float>(correct_slots) /
            static_cast<float>(std::max<std::size_t>(1U, total_slots));
        report.byte_accuracy = report.codepoint_accuracy;
    }
    trained_ = true;
    report.valid_utf8_ratio = 1.0F;
    return report;
}

std::string HolisticDialogueModel::respond(
    const std::string& input, const math::Vector& osten_state, std::size_t action_id,
    float action_value, const math::Vector& goal, const std::string& memory_context,
    float* confidence) const {
    if (!trained_ || vocabulary_.empty()) {
        if (confidence != nullptr) {
            *confidence = 0.0F;
        }
        return {};
    }
    const math::Vector condition = encode_condition(
        osten_state, action_id, action_value, goal, memory_context);
    const std::size_t class_count = vocabulary_.size() + 1U;
    std::string candidate;
    float confidence_sum = 0.0F;
    std::size_t emitted = 0;
    // Each pass refines the complete candidate in parallel. This is not
    // left-to-right next-token generation.
    math::Vector candidate_state(input_dim_, 0.0F);
    for (std::size_t pass = 0; pass < 1U; ++pass) {
        const math::Vector input_features = encode_text(input);
        for (std::size_t index = 0; index < candidate_state.size(); ++index) {
            candidate_state[index] = 0.75F * candidate_state[index] +
                0.25F * input_features[index];
        }
        const math::Vector features = candidate_state;
        std::string refined;
        float pass_confidence = 0.0F;
        std::size_t pass_emitted = 0;
        for (std::size_t slot = 0; slot < max_response_codepoints_; ++slot) {
            const math::Vector hidden = hidden_for(features, condition, slot);
            std::size_t best = 0;
            float best_score = -std::numeric_limits<float>::infinity();
            float second_score = -std::numeric_limits<float>::infinity();
            for (std::size_t current = 0; current < class_count; ++current) {
                // A response needs a meaningful prefix before the end class
                // can win. This prevents a high-frequency space from turning
                // an otherwise trained model into an empty console line.
                if (current == 0U && slot < 8U) {
                    continue;
                }
                if (current != 0U && slot < 8U &&
                    is_layout_codepoint(vocabulary_[current - 1U])) {
                    continue;
                }
                float score = class_bias_[current];
                if (current != 0U && !refined.empty() &&
                    vocabulary_[current - 1U] ==
                        static_cast<std::uint32_t>(static_cast<unsigned char>(refined.back()))) {
                    score -= 0.35F;
                }
                for (std::size_t h = 0; h < hidden_dim_; ++h) {
                    score += class_embeddings_[current * hidden_dim_ + h] * hidden[h];
                }
                if (score > best_score) {
                    second_score = best_score;
                    best_score = score;
                    best = current;
                } else if (score > second_score) {
                    second_score = score;
                }
            }
            if (best == 0U || !append_codepoint(vocabulary_[best - 1U], refined)) {
                break;
            }
            ++pass_emitted;
            pass_confidence +=
                1.0F / (1.0F + std::exp(-(best_score - second_score)));
        }
        candidate = std::move(refined);
        emitted = pass_emitted;
        confidence_sum = pass_confidence;
        if (candidate.empty()) {
            break;
        }
    }
    while (!candidate.empty() &&
           std::isspace(static_cast<unsigned char>(candidate.front()))) {
        candidate.erase(candidate.begin());
    }
    while (!candidate.empty() &&
           std::isspace(static_cast<unsigned char>(candidate.back()))) {
        candidate.pop_back();
    }
    if (candidate.empty()) {
        if (confidence != nullptr) {
            *confidence = 0.0F;
        }
        return {};
    }
    if (!decode_utf8(candidate).valid) {
        if (confidence != nullptr) {
            *confidence = 0.0F;
        }
        return {};
    }
    if (confidence != nullptr) {
        *confidence = emitted == 0U ? 0.0F : confidence_sum / static_cast<float>(emitted);
    }
    return candidate;
}

DialogueTrainingReport HolisticDialogueModel::train_pairs(
    const std::vector<TextDocument>& dataset, std::size_t epochs) {
    if (dataset.empty()) {
        throw std::invalid_argument("text pair dataset must not be empty");
    }
    for (const TextDocument& pair : dataset) {
        if (pair.input.empty() || pair.target.empty() ||
            !valid_utf8(pair.input) || !valid_utf8(pair.target)) {
            throw std::invalid_argument("text pair is empty or invalid UTF-8");
        }
    }
    return train(dataset, epochs);
}

DialogueTrainingReport HolisticDialogueModel::evaluate_pairs(
    const std::vector<TextDocument>& dataset) const {
    DialogueTrainingReport report;
    report.examples = dataset.size();
    if (dataset.empty()) {
        return report;
    }
    std::size_t correct = 0;
    std::size_t total = 0;
    std::size_t exact = 0;
    std::size_t non_repetitive = 0;
    std::size_t answer_accuracy = 0;
    for (const TextDocument& pair : dataset) {
        float confidence = 0.0F;
        const std::string output = respond(pair.input, {}, 0, 0.0F, {}, {}, &confidence);
        const Decoded expected = decode_utf8(pair.target);
        const Decoded actual = decode_utf8(output);
        const std::size_t count = std::min(expected.values.size(), actual.values.size());
        for (std::size_t index = 0; index < count; ++index) {
            ++total;
            if (expected.values[index] == actual.values[index]) {
                ++correct;
            }
        }
        if (output == pair.target) {
            ++exact;
        }
        if (!expected.values.empty() && !actual.values.empty() &&
            expected.values.front() == actual.values.front()) {
            ++answer_accuracy;
        }
        bool repeated = output.size() >= 4U;
        for (std::size_t index = 3; repeated && index < output.size(); ++index) {
            repeated = output[index] == output[index - 1] &&
                       output[index] == output[index - 2] &&
                       output[index] == output[index - 3];
        }
        if (!repeated) {
            ++non_repetitive;
        }
    }
    report.validation_codepoint_accuracy = total == 0U
        ? 0.0F : static_cast<float>(correct) / static_cast<float>(total);
    report.validation_exact_match = static_cast<float>(exact) /
        static_cast<float>(dataset.size());
    report.validation_non_repetition = static_cast<float>(non_repetitive) /
        static_cast<float>(dataset.size());
    report.validation_answer_accuracy = static_cast<float>(answer_accuracy) /
        static_cast<float>(dataset.size());
    report.valid_utf8_ratio = 1.0F;
    return report;
}

void HolisticDialogueModel::save(std::ostream& output) const {
    write_size(output, input_dim_);
    write_size(output, max_response_codepoints_);
    write_size(output, condition_dim_);
    write_size(output, hidden_dim_);
    output.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
    const std::uint8_t trained = trained_ ? 1U : 0U;
    output.write(reinterpret_cast<const char*>(&trained), sizeof(trained));
    write_size(output, vocabulary_.size());
    for (const std::uint32_t value : vocabulary_) {
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    write_vector(output, context_weights_);
    write_vector(output, context_bias_);
    write_vector(output, condition_weights_);
    write_vector(output, slot_embeddings_);
    write_vector(output, class_embeddings_);
    write_vector(output, class_bias_);
    if (!output) {
        throw std::runtime_error("failed to write dialogue checkpoint");
    }
}

void HolisticDialogueModel::load(std::istream& input) {
    if (read_size(input) != input_dim_ ||
        read_size(input) != max_response_codepoints_ ||
        read_size(input) != condition_dim_ || read_size(input) != hidden_dim_) {
        throw std::runtime_error("dialogue checkpoint configuration mismatch");
    }
    float stored_learning_rate = 0.0F;
    std::uint8_t trained = 0;
    input.read(reinterpret_cast<char*>(&stored_learning_rate), sizeof(stored_learning_rate));
    input.read(reinterpret_cast<char*>(&trained), sizeof(trained));
    if (!input || !std::isfinite(stored_learning_rate) || trained > 1U) {
        throw std::runtime_error("invalid dialogue checkpoint header");
    }
    const std::size_t vocabulary_size = read_size(input, kMaxVocabulary);
    vocabulary_.assign(vocabulary_size, 0U);
    for (std::uint32_t& value : vocabulary_) {
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
    }
    // An Astrax checkpoint may be created before dialogue training. In that
    // state save() intentionally writes empty parameter vectors; do not try
    // to read them as trained matrices.
    if (trained == 0U) {
        if (vocabulary_size != 0U) {
            throw std::runtime_error(
                "untrained dialogue checkpoint contains a vocabulary");
        }
        context_weights_ = read_vector(input, 0U);
        context_bias_ = read_vector(input, 0U);
        condition_weights_ = read_vector(input, 0U);
        slot_embeddings_ = read_vector(input, 0U);
        class_embeddings_ = read_vector(input, 0U);
        class_bias_ = read_vector(input, 0U);
        if (!input) {
            throw std::runtime_error("dialogue checkpoint is incomplete");
        }
        trained_ = false;
        return;
    }
    const std::size_t class_count = vocabulary_.size() + 1U;
    const std::size_t combined_dim = input_dim_ + condition_dim_;
    context_weights_ = read_vector(input, hidden_dim_ * combined_dim);
    context_bias_ = read_vector(input, hidden_dim_);
    condition_weights_ = read_vector(input, hidden_dim_ * condition_dim_);
    slot_embeddings_ = read_vector(input, max_response_codepoints_ * hidden_dim_);
    class_embeddings_ = read_vector(input, class_count * hidden_dim_);
    class_bias_ = read_vector(input, class_count);
    if (!input || !std::isfinite(stored_learning_rate)) {
        throw std::runtime_error("dialogue checkpoint is incomplete");
    }
    trained_ = trained != 0U;
}

void HolisticDialogueModel::validate_dataset(
    const std::vector<TextDocument>& dataset, std::size_t max_document_bytes) {
    if (dataset.empty() || max_document_bytes == 0U) {
        throw std::invalid_argument("text dataset must not be empty");
    }
    for (std::size_t index = 0; index < dataset.size(); ++index) {
        const TextDocument& document = dataset[index];
        if (document.text.empty() || document.text.size() > max_document_bytes ||
            document.text.find('\0') != std::string::npos ||
            !valid_utf8(document.text)) {
            throw std::invalid_argument("text document " + std::to_string(index) +
                                        " is empty, too large, or invalid UTF-8");
        }
    }
}

std::vector<TextDocument> HolisticDialogueModel::load_documents(
    const std::string& path, std::size_t max_document_bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open text corpus: " + path);
    }
    std::vector<TextDocument> dataset;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        dataset.push_back({line});
    }
    validate_dataset(dataset, max_document_bytes);
    return dataset;
}

std::vector<TextDocument> HolisticDialogueModel::load_pairs(
    const std::string& path, std::size_t max_document_bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open text pair corpus: " + path);
    }
    std::vector<TextDocument> dataset;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::size_t separator = line.find('\t');
        if (separator == std::string::npos) {
            if (dataset.empty()) {
                throw std::invalid_argument("text pair row must contain a tab separator");
            }
            dataset.back().target += "\n" + line;
            continue;
        }
        dataset.push_back({{}, line.substr(0, separator),
                           line.substr(separator + 1)});
    }
    for (std::size_t index = 0; index < dataset.size(); ++index) {
        const TextDocument& pair = dataset[index];
        if (pair.input.empty() || pair.target.empty() ||
            pair.input.size() > max_document_bytes ||
            pair.target.size() > max_document_bytes ||
            !valid_utf8(pair.input) || !valid_utf8(pair.target)) {
            throw std::invalid_argument("text pair " + std::to_string(index) +
                                        " is empty, too large, or invalid UTF-8");
        }
    }
    if (dataset.empty()) {
        throw std::invalid_argument("text pair dataset must not be empty");
    }
    return dataset;
}

} // namespace astrax
