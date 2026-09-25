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
#include <random>
#include <sstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "astrax/charset.hpp"

namespace astrax {
namespace {

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
            (codepoint >= 0x20U || codepoint == 10U ||
             codepoint == 13U || codepoint == 9U)) {
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
      condition_dim_(condition_dim) {
    if (input_dim_ == 0 || max_response_codepoints_ == 0 || condition_dim_ == 0 ||
        !std::isfinite(learning_rate_) || learning_rate_ <= 0.0F) {
        throw std::invalid_argument("invalid holistic dialogue configuration");
    }
    if (rounds_ == 0U) {
        throw std::invalid_argument("holistic dialogue requires at least one round");
    }
    high_count_ = charset::kHighCount;
    low_count_ = charset::kLowCount;
}

math::Vector HolisticDialogueModel::encode_text(const std::string& input) const {
    // Prefer the learned subword encoder when attached: it gives the input a
    // shared representation across synonyms, morphology, and mixed scripts.
    if (has_encoder_ && !encoder_.empty() && input_dim_ > 0) {
        const std::vector<std::uint32_t> ids = encoder_.encode_ids(input);
        math::Vector pooled = embeddings_.encode(ids);
        math::Vector projected(input_dim_, 0.0F);
        for (std::size_t index = 0; index < pooled.size(); ++index) {
            projected[index % projected.size()] += pooled[index];
        }
        const float length = math::norm(projected);
        if (length > 1.0F) {
            for (float& value : projected) {
                value /= length;
            }
        }
        return projected;
    }
    math::Vector result(input_dim_, 0.0F);
    math::Vector reverse_state(input_dim_, 0.0F);
    const Decoded decoded = decode_utf8(input);
    const auto& values = decoded.values;
    if (values.empty()) {
        return result;
    }
    std::size_t events = 0;
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

math::Vector HolisticDialogueModel::softmax(const math::Vector& logits) {
    math::Vector result(logits.size(), 0.0F);
    if (logits.empty()) {
        return result;
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (const float value : logits) {
        maximum = std::max(maximum, value);
    }
    float normalizer = 0.0F;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        result[index] = std::exp(logits[index] - maximum);
        normalizer += result[index];
    }
    if (normalizer <= 1.0e-12F) {
        return math::Vector(logits.size(), 1.0F / static_cast<float>(logits.size()));
    }
    for (float& value : result) {
        value /= normalizer;
    }
    return result;
}

void HolisticDialogueModel::forward_pass(
    const math::Vector& features, const math::Vector& condition,
    const math::Vector& aggregate, std::size_t round,
    std::vector<SlotLogits>& logits) const {
    logits.assign(max_response_codepoints_, SlotLogits{
        math::Vector(high_count_, 0.0F), math::Vector(low_count_, 0.0F)});
    const std::size_t agg = std::min(aggregate.size(), agg_dim_);
    // Precompute the slot-independent projection once per pass. Input,
    // condition, aggregate, and bias are identical for every slot; only the
    // slot and round embeddings vary. This keeps the denoising loop affordable
    // at research scale.
    math::Vector base(hidden_dim_, 0.0F);
    for (std::size_t h = 0; h < hidden_dim_; ++h) {
        float value = hidden_bias_[h];
        const std::size_t in_row = h * input_dim_;
        for (std::size_t column = 0; column < input_dim_; ++column) {
            value += in_weights_[in_row + column] * features[column];
        }
        const std::size_t cond_row = h * condition_dim_;
        for (std::size_t column = 0; column < condition_dim_; ++column) {
            value += cond_weights_[cond_row + column] * condition[column];
        }
        const std::size_t agg_row = h * agg_dim_;
        for (std::size_t column = 0; column < agg; ++column) {
            value += agg_weights_[agg_row + column] * aggregate[column];
        }
        base[h] = value;
    }
    for (std::size_t slot = 0; slot < max_response_codepoints_; ++slot) {
        math::Vector hidden(hidden_dim_, 0.0F);
        for (std::size_t h = 0; h < hidden_dim_; ++h) {
            float value = base[h];
            const std::size_t slot_row = h * slot_dim_;
            for (std::size_t column = 0; column < slot_dim_; ++column) {
                value += slot_weights_[slot_row + column] *
                    slot_embedding_table_[slot * slot_dim_ + column];
            }
            const std::size_t round_row = h * round_dim_;
            for (std::size_t column = 0; column < round_dim_; ++column) {
                value += round_weights_[round_row + column] *
                    round_embedding_table_[round * round_dim_ + column];
            }
            hidden[h] = std::tanh(value);
        }
        SlotLogits& slot_logits = logits[slot];
        for (std::size_t c = 0; c < high_count_; ++c) {
            float value = out_high_bias_[c];
            const std::size_t row = c * hidden_dim_;
            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                value += out_high_weights_[row + h] * hidden[h];
            }
            slot_logits.high[c] = value;
        }
        for (std::size_t c = 0; c < low_count_; ++c) {
            float value = out_low_bias_[c];
            const std::size_t row = c * hidden_dim_;
            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                value += out_low_weights_[row + h] * hidden[h];
            }
            slot_logits.low[c] = value;
        }
    }
}

math::Vector HolisticDialogueModel::aggregate_slots(
    const std::vector<SlotLogits>& logits) const {
    math::Vector aggregate(agg_dim_, 0.0F);
    if (logits.empty()) {
        return aggregate;
    }
    for (std::size_t slot = 0; slot < logits.size(); ++slot) {
        const math::Vector high_probabilities = softmax(logits[slot].high);
        for (std::size_t c = 0; c < high_probabilities.size(); ++c) {
            const std::uint64_t hash = mix_hash(
                static_cast<std::uint64_t>(slot) * 0x9e3779b97f4a7c15ULL +
                static_cast<std::uint64_t>(c) * 0xbf58476d1ce4e5b9ULL);
            aggregate[hash % agg_dim_] += high_probabilities[c];
        }
        const math::Vector low_probabilities = softmax(logits[slot].low);
        for (std::size_t c = 0; c < low_probabilities.size(); ++c) {
            const std::uint64_t hash = mix_hash(
                static_cast<std::uint64_t>(slot) * 0x94d049bb133111ebULL +
                static_cast<std::uint64_t>(c) * 0x2545f4914f6cdd1dULL + 0x9e3779b9ULL);
            aggregate[hash % agg_dim_] += low_probabilities[c];
        }
    }
    const float length = math::norm(aggregate);
    if (length > 1.0F) {
        for (float& value : aggregate) {
            value /= length;
        }
    }
    return aggregate;
}

float HolisticDialogueModel::train_pass(
    const math::Vector& features, const math::Vector& condition,
    const math::Vector& aggregate, std::size_t round,
    const std::vector<std::uint32_t>& targets, const std::vector<char>& active) {
    const std::size_t agg = std::min(aggregate.size(), agg_dim_);
    float total_loss = 0.0F;
    std::size_t active_count = 0;
    // Slot-independent projection, computed once per pass.
    math::Vector base(hidden_dim_, 0.0F);
    for (std::size_t h = 0; h < hidden_dim_; ++h) {
        float value = hidden_bias_[h];
        const std::size_t in_row = h * input_dim_;
        for (std::size_t column = 0; column < input_dim_; ++column) {
            value += in_weights_[in_row + column] * features[column];
        }
        const std::size_t cond_row = h * condition_dim_;
        for (std::size_t column = 0; column < condition_dim_; ++column) {
            value += cond_weights_[cond_row + column] * condition[column];
        }
        const std::size_t agg_row = h * agg_dim_;
        for (std::size_t column = 0; column < agg; ++column) {
            value += agg_weights_[agg_row + column] * aggregate[column];
        }
        base[h] = value;
    }
    for (std::size_t slot = 0; slot < max_response_codepoints_; ++slot) {
        if (slot >= active.size() || active[slot] == 0) {
            continue;
        }
        const std::uint32_t codepoint = targets[slot];
        const std::size_t target_high = charset::high_of(codepoint);
        const std::size_t target_low = charset::low_of(codepoint);
        math::Vector hidden(hidden_dim_, 0.0F);
        for (std::size_t h = 0; h < hidden_dim_; ++h) {
            float value = base[h];
            const std::size_t slot_row = h * slot_dim_;
            for (std::size_t column = 0; column < slot_dim_; ++column) {
                value += slot_weights_[slot_row + column] *
                    slot_embedding_table_[slot * slot_dim_ + column];
            }
            const std::size_t round_row = h * round_dim_;
            for (std::size_t column = 0; column < round_dim_; ++column) {
                value += round_weights_[round_row + column] *
                    round_embedding_table_[round * round_dim_ + column];
            }
            hidden[h] = std::tanh(value);
        }
        math::Vector high_logits(high_count_, 0.0F);
        float high_maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < high_count_; ++c) {
            float value = out_high_bias_[c];
            const std::size_t row = c * hidden_dim_;
            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                value += out_high_weights_[row + h] * hidden[h];
            }
            high_logits[c] = value;
            high_maximum = std::max(high_maximum, value);
        }
        float high_normalizer = 0.0F;
        for (const float value : high_logits) {
            high_normalizer += std::exp(value - high_maximum);
        }
        high_normalizer = std::max(high_normalizer, 1.0e-12F);
        math::Vector low_logits(low_count_, 0.0F);
        float low_maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < low_count_; ++c) {
            float value = out_low_bias_[c];
            const std::size_t row = c * hidden_dim_;
            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                value += out_low_weights_[row + h] * hidden[h];
            }
            low_logits[c] = value;
            low_maximum = std::max(low_maximum, value);
        }
        float low_normalizer = 0.0F;
        for (const float value : low_logits) {
            low_normalizer += std::exp(value - low_maximum);
        }
        low_normalizer = std::max(low_normalizer, 1.0e-12F);
        total_loss += -(high_logits[target_high] - high_maximum -
                        std::log(high_normalizer));
        total_loss += -(low_logits[target_low] - low_maximum -
                        std::log(low_normalizer));
        ++active_count;

        math::Vector hidden_gradient(hidden_dim_, 0.0F);
        for (std::size_t c = 0; c < high_count_; ++c) {
            const float probability = std::exp(high_logits[c] - high_maximum) /
                                      high_normalizer;
            const float g = (probability - (c == target_high ? 1.0F : 0.0F)) *
                            learning_rate_;
            out_high_bias_[c] -= g;
            const std::size_t row = c * hidden_dim_;
            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                hidden_gradient[h] += g * out_high_weights_[row + h];
                out_high_weights_[row + h] -= g * hidden[h];
            }
        }
        for (std::size_t c = 0; c < low_count_; ++c) {
            const float probability = std::exp(low_logits[c] - low_maximum) /
                                      low_normalizer;
            const float g = (probability - (c == target_low ? 1.0F : 0.0F)) *
                            learning_rate_;
            out_low_bias_[c] -= g;
            const std::size_t row = c * hidden_dim_;
            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                hidden_gradient[h] += g * out_low_weights_[row + h];
                out_low_weights_[row + h] -= g * hidden[h];
            }
        }
        math::Vector pre_gradient(hidden_dim_, 0.0F);
        for (std::size_t h = 0; h < hidden_dim_; ++h) {
            pre_gradient[h] = hidden_gradient[h] * (1.0F - hidden[h] * hidden[h]);
        }
        for (std::size_t h = 0; h < hidden_dim_; ++h) {
            const float g = pre_gradient[h];
            hidden_bias_[h] -= g;
            const std::size_t in_row = h * input_dim_;
            for (std::size_t column = 0; column < input_dim_; ++column) {
                in_weights_[in_row + column] -= g * features[column];
            }
            const std::size_t cond_row = h * condition_dim_;
            for (std::size_t column = 0; column < condition_dim_; ++column) {
                cond_weights_[cond_row + column] -= g * condition[column];
            }
            const std::size_t slot_row = h * slot_dim_;
            for (std::size_t column = 0; column < slot_dim_; ++column) {
                const float embedding = slot_embedding_table_[slot * slot_dim_ + column];
                const float weight = slot_weights_[slot_row + column];
                slot_weights_[slot_row + column] -= g * embedding;
                slot_embedding_table_[slot * slot_dim_ + column] -= g * weight;
            }
            const std::size_t round_row = h * round_dim_;
            for (std::size_t column = 0; column < round_dim_; ++column) {
                const float embedding = round_embedding_table_[round * round_dim_ + column];
                const float weight = round_weights_[round_row + column];
                round_weights_[round_row + column] -= g * embedding;
                round_embedding_table_[round * round_dim_ + column] -= g * weight;
            }
            const std::size_t agg_row = h * agg_dim_;
            for (std::size_t column = 0; column < agg; ++column) {
                agg_weights_[agg_row + column] -= g * aggregate[column];
            }
        }
    }
    return active_count == 0U ? 0.0F : total_loss / static_cast<float>(active_count);
}
HolisticDialogueModel::Replica HolisticDialogueModel::capture() const {
    Replica replica;
    replica.in_weights = in_weights_;
    replica.cond_weights = cond_weights_;
    replica.slot_weights = slot_weights_;
    replica.round_weights = round_weights_;
    replica.agg_weights = agg_weights_;
    replica.hidden_bias = hidden_bias_;
    replica.out_high_weights = out_high_weights_;
    replica.out_high_bias = out_high_bias_;
    replica.out_low_weights = out_low_weights_;
    replica.out_low_bias = out_low_bias_;
    replica.slot_embedding_table = slot_embedding_table_;
    replica.round_embedding_table = round_embedding_table_;
    return replica;
}

void HolisticDialogueModel::install(const Replica& replica) {
    in_weights_ = replica.in_weights;
    cond_weights_ = replica.cond_weights;
    slot_weights_ = replica.slot_weights;
    round_weights_ = replica.round_weights;
    agg_weights_ = replica.agg_weights;
    hidden_bias_ = replica.hidden_bias;
    out_high_weights_ = replica.out_high_weights;
    out_high_bias_ = replica.out_high_bias;
    out_low_weights_ = replica.out_low_weights;
    out_low_bias_ = replica.out_low_bias;
    slot_embedding_table_ = replica.slot_embedding_table;
    round_embedding_table_ = replica.round_embedding_table;
}

void HolisticDialogueModel::accumulate(const Replica& replica) {
    const auto add = [](math::Vector& target, const math::Vector& source) {
        for (std::size_t index = 0; index < target.size(); ++index) {
            target[index] += source[index];
        }
    };
    add(in_weights_, replica.in_weights);
    add(cond_weights_, replica.cond_weights);
    add(slot_weights_, replica.slot_weights);
    add(round_weights_, replica.round_weights);
    add(agg_weights_, replica.agg_weights);
    add(hidden_bias_, replica.hidden_bias);
    add(out_high_weights_, replica.out_high_weights);
    add(out_high_bias_, replica.out_high_bias);
    add(out_low_weights_, replica.out_low_weights);
    add(out_low_bias_, replica.out_low_bias);
    add(slot_embedding_table_, replica.slot_embedding_table);
    add(round_embedding_table_, replica.round_embedding_table);
}

void HolisticDialogueModel::train_range(
    const std::vector<TextDocument>& dataset,
    const std::vector<math::Vector>& conditions,
    std::size_t begin, std::size_t end, float& loss,
    std::size_t& total_slots, std::size_t& correct_slots) {
    loss = 0.0F;
    total_slots = 0;
    correct_slots = 0;
    std::vector<SlotLogits> logits;
    for (std::size_t document_index = begin; document_index < end;
         ++document_index) {
        const TextDocument& document = dataset[document_index];
        const std::string& source_text = document.input.empty()
            ? document.text : document.input;
        const std::string& target_text = document.target.empty()
            ? document.text : document.target;
        const Decoded target_values = decode_utf8(target_text);
        if (target_values.values.empty()) {
            continue;
        }
        const math::Vector features = encode_text(source_text);
        const math::Vector condition = conditions.empty()
            ? encode_condition({}, 0U, 0.0F, {}, {})
            : conditions[document_index];
        const std::size_t length = std::min(
            target_values.values.size(), max_response_codepoints_);
        std::vector<std::uint32_t> targets(max_response_codepoints_, 0U);
        std::vector<char> active(max_response_codepoints_, 0);
        for (std::size_t slot = 0; slot < length; ++slot) {
            targets[slot] = target_values.values[slot];
            active[slot] = 1;
        }
        math::Vector aggregate(agg_dim_, 0.0F);
        for (std::size_t round = 0; round < rounds_; ++round) {
            loss += train_pass(features, condition, aggregate, round, targets,
                               active);
            forward_pass(features, condition, aggregate, round, logits);
            for (std::size_t slot = 0; slot < length; ++slot) {
                ++total_slots;
                const std::uint32_t predicted = charset::combine(
                    static_cast<std::uint32_t>(math::argmax(logits[slot].high)),
                    static_cast<std::uint32_t>(math::argmax(logits[slot].low)));
                if (predicted == targets[slot]) {
                    ++correct_slots;
                }
            }
            aggregate = aggregate_slots(logits);
        }
    }
}

DialogueTrainingReport HolisticDialogueModel::train(
    const std::vector<TextDocument>& dataset, std::size_t epochs,
    const std::vector<math::Vector>& conditions) {
    if (!conditions.empty() && conditions.size() != dataset.size()) {
        throw std::invalid_argument(
            "condition count must match the text training document count");
    }
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

    // The model is told the complete Unicode codepoint space and selects any
    // codepoint itself. There is no data-derived vocabulary: the output layer
    // is factorized into a high part and a low part over the full range.
    const bool fresh = in_weights_.empty();
    if (fresh) {
        in_weights_.assign(hidden_dim_ * input_dim_, 0.0F);
        cond_weights_.assign(hidden_dim_ * condition_dim_, 0.0F);
        slot_weights_.assign(hidden_dim_ * slot_dim_, 0.0F);
        round_weights_.assign(hidden_dim_ * round_dim_, 0.0F);
        agg_weights_.assign(hidden_dim_ * agg_dim_, 0.0F);
        hidden_bias_.assign(hidden_dim_, 0.0F);
        out_high_weights_.assign(high_count_ * hidden_dim_, 0.0F);
        out_high_bias_.assign(high_count_, 0.0F);
        out_low_weights_.assign(low_count_ * hidden_dim_, 0.0F);
        out_low_bias_.assign(low_count_, 0.0F);
        slot_embedding_table_.assign(max_response_codepoints_ * slot_dim_, 0.0F);
        round_embedding_table_.assign(rounds_ * round_dim_, 0.0F);
        for (std::size_t index = 0; index < in_weights_.size(); ++index) {
            const float phase = static_cast<float>((index * 17U + 11U) % 211U) / 211.0F;
            in_weights_[index] = 0.03F * std::sin(phase * 6.2831853F);
        }
        for (std::size_t index = 0; index < cond_weights_.size(); ++index) {
            const float phase = static_cast<float>((index * 23U + 5U) % 197U) / 197.0F;
            cond_weights_[index] = 0.03F * std::cos(phase * 6.2831853F);
        }
        for (std::size_t index = 0; index < slot_embedding_table_.size(); ++index) {
            const float phase = static_cast<float>((index * 29U + 7U) % 173U) / 173.0F;
            slot_embedding_table_[index] = 0.1F * std::cos(phase * 6.2831853F);
        }
        for (std::size_t index = 0; index < round_embedding_table_.size(); ++index) {
            const float phase = static_cast<float>((index * 13U + 3U) % 149U) / 149.0F;
            round_embedding_table_[index] = 0.1F * std::sin(phase * 6.2831853F);
        }
        for (std::size_t index = 0; index < slot_weights_.size(); ++index) {
            const float phase = static_cast<float>((index * 31U + 2U) % 181U) / 181.0F;
            slot_weights_[index] = 0.05F * std::sin(phase * 6.2831853F);
        }
        for (std::size_t index = 0; index < round_weights_.size(); ++index) {
            const float phase = static_cast<float>((index * 19U + 9U) % 167U) / 167.0F;
            round_weights_[index] = 0.05F * std::cos(phase * 6.2831853F);
        }
        for (std::size_t index = 0; index < agg_weights_.size(); ++index) {
            const float phase = static_cast<float>((index * 37U + 4U) % 193U) / 193.0F;
            agg_weights_[index] = 0.04F * std::sin(phase * 6.2831853F);
        }
        for (std::size_t index = 0; index < out_high_weights_.size(); ++index) {
            const float phase = static_cast<float>((index * 41U) % 199U) / 199.0F;
            out_high_weights_[index] = 0.08F * std::sin(phase * 6.2831853F);
        }
        for (std::size_t index = 0; index < out_low_weights_.size(); ++index) {
            const float phase = static_cast<float>((index * 43U + 6U) % 211U) / 211.0F;
            out_low_weights_[index] = 0.05F * std::sin(phase * 6.2831853F);
        }
    }

    const std::size_t worker_count = std::max<std::size_t>(
        1U, std::min<std::size_t>(dataset.size(),
                                  std::thread::hardware_concurrency()));

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        // Data-parallel epoch: every worker trains a private replica from the
        // same snapshot over a distinct document range, and the replicas are
        // averaged back. This keeps the learned parameters identical in kind
        // to single-threaded SGD while using all cores.
        const Replica snapshot = capture();
        std::vector<Replica> replicas(worker_count);
        std::vector<float> losses(worker_count, 0.0F);
        std::vector<std::size_t> slot_counts(worker_count, 0U);
        std::vector<std::size_t> correct_counts(worker_count, 0U);
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        const std::size_t chunk = dataset.size() / worker_count;
        const std::size_t remainder = dataset.size() % worker_count;
        std::size_t cursor = 0;
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            const std::size_t begin = cursor;
            const std::size_t length = chunk + (worker < remainder ? 1U : 0U);
            cursor += length;
            const std::size_t end = cursor;
            workers.emplace_back([this, &dataset, &conditions, &snapshot,
                                  &replicas, &losses, &slot_counts,
                                  &correct_counts, begin, end, worker]() {
                // Each worker owns a private model replica so no two threads
                // ever write the same parameter memory. The replica starts
                // from this epoch's snapshot and is merged after the join.
                HolisticDialogueModel local = *this;
                local.install(snapshot);
                local.train_range(dataset, conditions, begin, end,
                                  losses[worker], slot_counts[worker],
                                  correct_counts[worker]);
                replicas[worker] = local.capture();
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
        // Average the replicas back into the live parameters. Install the
        // first replica, then accumulate the rest, then scale by 1/worker.
        install(replicas.front());
        for (std::size_t worker = 1; worker < worker_count; ++worker) {
            accumulate(replicas[worker]);
        }
        const float inverse = 1.0F / static_cast<float>(worker_count);
        const auto scale = [inverse](math::Vector& values) {
            for (float& value : values) {
                value *= inverse;
            }
        };
        scale(in_weights_);
        scale(cond_weights_);
        scale(slot_weights_);
        scale(round_weights_);
        scale(agg_weights_);
        scale(hidden_bias_);
        scale(out_high_weights_);
        scale(out_high_bias_);
        scale(out_low_weights_);
        scale(out_low_bias_);
        scale(slot_embedding_table_);
        scale(round_embedding_table_);

        float total_loss = 0.0F;
        std::size_t total_slots = 0;
        std::size_t correct_slots = 0;
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            total_loss += losses[worker];
            total_slots += slot_counts[worker];
            correct_slots += correct_counts[worker];
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
    if (!trained_) {
        if (confidence != nullptr) {
            *confidence = 0.0F;
        }
        return {};
    }
    const math::Vector condition = encode_condition(
        osten_state, action_id, action_value, goal, memory_context);
    const math::Vector features = encode_text(input);
    std::vector<SlotLogits> logits;
    math::Vector aggregate(agg_dim_, 0.0F);
    for (std::size_t round = 0; round < rounds_; ++round) {
        forward_pass(features, condition, aggregate, round, logits);
        aggregate = aggregate_slots(logits);
    }

    std::string candidate;
    float confidence_sum = 0.0F;
    std::size_t emitted = 0;
    for (std::size_t slot = 0; slot < max_response_codepoints_; ++slot) {
        const math::Vector high_probabilities = softmax(logits[slot].high);
        const math::Vector low_probabilities = softmax(logits[slot].low);
        // The high and low factors are selected independently. Tracking a
        // single shared score would let the larger high distribution suppress
        // every low candidate, which collapsed the low factor to class zero.
        std::size_t best_high = 0;
        std::size_t best_low = 0;
        float best_high_score = -std::numeric_limits<float>::infinity();
        float second_high_score = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < high_count_; ++c) {
            const float score = high_probabilities[c];
            if (score > best_high_score) {
                second_high_score = best_high_score;
                best_high_score = score;
                best_high = c;
            } else if (score > second_high_score) {
                second_high_score = score;
            }
        }
        float best_low_score = -std::numeric_limits<float>::infinity();
        float second_low_score = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < low_count_; ++c) {
            const float score = low_probabilities[c];
            if (score > best_low_score) {
                second_low_score = best_low_score;
                best_low_score = score;
                best_low = c;
            } else if (score > second_low_score) {
                second_low_score = score;
            }
        }
        // The reserved codepoint 0 (high 0, low 0) ends the response only when
        // both factors independently choose it.
        const bool stop = best_high == 0U && best_low == 0U && slot >= 2U;
        const std::uint32_t codepoint = charset::combine(
            static_cast<std::uint32_t>(best_high),
            static_cast<std::uint32_t>(best_low));
        if (stop || !charset::is_valid_codepoint(codepoint) ||
            !append_codepoint(codepoint, candidate)) {
            break;
        }
        ++emitted;
        const float margin = (best_high_score - second_high_score) +
                             (best_low_score - second_low_score);
        confidence_sum += 1.0F / (1.0F + std::exp(-margin * 4.0F));
    }
    while (!candidate.empty() &&
           std::isspace(static_cast<unsigned char>(candidate.front()))) {
        candidate.erase(candidate.begin());
    }
    while (!candidate.empty() &&
           std::isspace(static_cast<unsigned char>(candidate.back()))) {
        candidate.pop_back();
    }
    if (candidate.empty() || !decode_utf8(candidate).valid) {
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
    const std::vector<TextDocument>& dataset, std::size_t epochs,
    const std::vector<math::Vector>& conditions) {
    if (dataset.empty()) {
        throw std::invalid_argument("text pair dataset must not be empty");
    }
    for (const TextDocument& pair : dataset) {
        if (pair.input.empty() || pair.target.empty() ||
            !valid_utf8(pair.input) || !valid_utf8(pair.target)) {
            throw std::invalid_argument("text pair is empty or invalid UTF-8");
        }
    }
    // Conditional fine-tuning continues from the current parameters. No target
    // text is stored for retrieval; the response is produced only by the
    // learned iterative denoiser.
    return train(dataset, epochs, conditions);
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
    write_size(output, slot_dim_);
    write_size(output, round_dim_);
    write_size(output, agg_dim_);
    write_size(output, rounds_);
    output.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
    const std::uint8_t trained = trained_ ? 1U : 0U;
    output.write(reinterpret_cast<const char*>(&trained), sizeof(trained));
    write_size(output, high_count_);
    write_size(output, low_count_);
    if (trained_ == 0U) {
        if (!output) {
            throw std::runtime_error("failed to write dialogue checkpoint");
        }
        return;
    }
    write_vector(output, in_weights_);
    write_vector(output, cond_weights_);
    write_vector(output, slot_weights_);
    write_vector(output, round_weights_);
    write_vector(output, agg_weights_);
    write_vector(output, hidden_bias_);
    write_vector(output, out_high_weights_);
    write_vector(output, out_high_bias_);
    write_vector(output, out_low_weights_);
    write_vector(output, out_low_bias_);
    write_vector(output, slot_embedding_table_);
    write_vector(output, round_embedding_table_);
    if (!output) {
        throw std::runtime_error("failed to write dialogue checkpoint");
    }
}

void HolisticDialogueModel::load(std::istream& input) {
    if (read_size(input) != input_dim_ ||
        read_size(input) != max_response_codepoints_ ||
        read_size(input) != condition_dim_ || read_size(input) != hidden_dim_ ||
        read_size(input) != slot_dim_ || read_size(input) != round_dim_ ||
        read_size(input) != agg_dim_ || read_size(input) != rounds_) {
        throw std::runtime_error("dialogue checkpoint configuration mismatch");
    }
    float stored_learning_rate = 0.0F;
    std::uint8_t trained = 0;
    input.read(reinterpret_cast<char*>(&stored_learning_rate), sizeof(stored_learning_rate));
    input.read(reinterpret_cast<char*>(&trained), sizeof(trained));
    if (!input || !std::isfinite(stored_learning_rate) || trained > 1U) {
        throw std::runtime_error("invalid dialogue checkpoint header");
    }
    if (read_size(input) != high_count_ || read_size(input) != low_count_) {
        throw std::runtime_error("dialogue checkpoint factor sizes do not match");
    }
    if (trained == 0U) {
        trained_ = false;
        return;
    }
    in_weights_ = read_vector(input, hidden_dim_ * input_dim_);
    cond_weights_ = read_vector(input, hidden_dim_ * condition_dim_);
    slot_weights_ = read_vector(input, hidden_dim_ * slot_dim_);
    round_weights_ = read_vector(input, hidden_dim_ * round_dim_);
    agg_weights_ = read_vector(input, hidden_dim_ * agg_dim_);
    hidden_bias_ = read_vector(input, hidden_dim_);
    out_high_weights_ = read_vector(input, high_count_ * hidden_dim_);
    out_high_bias_ = read_vector(input, high_count_);
    out_low_weights_ = read_vector(input, low_count_ * hidden_dim_);
    out_low_bias_ = read_vector(input, low_count_);
    slot_embedding_table_ = read_vector(input, max_response_codepoints_ * slot_dim_);
    round_embedding_table_ = read_vector(input, rounds_ * round_dim_);
    if (!input) {
        throw std::runtime_error("dialogue checkpoint is incomplete");
    }
    trained_ = true;
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

