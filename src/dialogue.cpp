#include "astrax/dialogue.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "astrax/architecture.hpp"

namespace astrax {
namespace {

constexpr std::size_t kByteClasses = 256;
constexpr std::size_t kMinNGram = 1;
constexpr std::size_t kMaxNGram = 4;

std::uint64_t fnv1a(const std::string& value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t mix_hash(std::uint64_t value) {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

math::Vector encode_text(const std::string& input, std::size_t dimension) {
    math::Vector result(dimension, 0.0F);
    if (input.empty()) {
        return result;
    }

    const std::string padded = "^" + input + "$";
    std::size_t count = 0;
    for (std::size_t n = kMinNGram; n <= kMaxNGram; ++n) {
        if (n > padded.size()) {
            break;
        }
        for (std::size_t index = 0; index + n <= padded.size(); ++index) {
            const std::string gram = padded.substr(index, n);
            const std::uint64_t hash = fnv1a(gram);
            const std::size_t bucket =
                static_cast<std::size_t>(hash % dimension);
            result[bucket] += (hash & 1ULL) != 0ULL ? 1.0F : -1.0F;
            ++count;
        }
    }

    // A whole-input signature makes the representation holistic while the
    // n-grams preserve useful generalization for unseen utterances. This is
    // learned association, not a retrieval table or a keyword branch.
    const std::uint64_t whole = fnv1a(input);
    for (std::size_t offset = 0; offset < 32; ++offset) {
        const std::uint64_t hash = mix_hash(whole + offset * 0x9e3779b97f4a7c15ULL);
        result[static_cast<std::size_t>(hash % dimension)] +=
            (hash & 1ULL) != 0ULL ? 3.0F : -3.0F;
        count += 3;
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(std::max<std::size_t>(1, count)));
    for (float& value : result) {
        value *= scale;
    }
    const float length = math::norm(result);
    if (length > 1.0F) {
        for (float& value : result) {
            value /= length;
        }
    }
    return result;
}

void write_size(std::ostream& output, std::size_t value) {
    const std::uint64_t stored = static_cast<std::uint64_t>(value);
    output.write(reinterpret_cast<const char*>(&stored), sizeof(stored));
}

std::size_t read_size(std::istream& input) {
    std::uint64_t stored = 0;
    input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
    if (!input || stored > 100000000ULL) {
        throw std::runtime_error("invalid dialogue checkpoint size");
    }
    return static_cast<std::size_t>(stored);
}

void write_vector(std::ostream& output, const math::Vector& value) {
    write_size(output, value.size());
    output.write(reinterpret_cast<const char*>(value.data()),
                 static_cast<std::streamsize>(value.size() * sizeof(float)));
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
    for (float value : result) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("dialogue checkpoint contains non-finite weights");
        }
    }
    return result;
}

std::vector<std::string> split_tab(const std::string& line) {
    const std::size_t separator = line.find('\t');
    if (separator == std::string::npos) {
        return {};
    }
    return {line.substr(0, separator), line.substr(separator + 1)};
}

} // namespace

HolisticDialogueModel::HolisticDialogueModel(std::size_t input_dim,
                                             std::size_t max_response_bytes,
                                             float learning_rate)
    : input_dim_(input_dim),
      max_response_bytes_(max_response_bytes),
      learning_rate_(learning_rate),
      weights_(max_response_bytes * kByteClasses * input_dim, 0.0F),
      bias_(max_response_bytes * kByteClasses, 0.0F) {
    if (input_dim_ == 0 || max_response_bytes_ == 0 ||
        !std::isfinite(learning_rate_) || learning_rate_ <= 0.0F) {
        throw std::invalid_argument("invalid holistic dialogue configuration");
    }
}

math::Vector HolisticDialogueModel::encode_input(const std::string& input) const {
    return encode_text(input, input_dim_);
}

std::size_t HolisticDialogueModel::weight_index(std::size_t slot,
                                                 std::size_t byte,
                                                 std::size_t feature) const noexcept {
    return (slot * kByteClasses + byte) * input_dim_ + feature;
}

DialogueTrainingReport HolisticDialogueModel::train(
    const std::vector<DialogueExample>& dataset, std::size_t epochs) {
    validate_dataset(dataset, max_response_bytes_);
    DialogueTrainingReport report;
    report.examples = dataset.size();
    report.epochs = epochs;
    if (epochs == 0) {
        return report;
    }

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        float loss = 0.0F;
        std::size_t correct = 0;
        std::size_t total = 0;
        std::size_t exact = 0;
        for (const DialogueExample& example : dataset) {
            const math::Vector features = encode_input(example.input);
            bool example_exact = true;
            const std::size_t training_slots =
                std::min(max_response_bytes_, example.response.size() + 1);
            for (std::size_t slot = 0; slot < training_slots; ++slot) {
                const std::size_t target =
                    slot < example.response.size()
                        ? static_cast<unsigned char>(example.response[slot])
                        : 0;
                std::array<float, kByteClasses> logits{};
                for (std::size_t byte = 0; byte < kByteClasses; ++byte) {
                    float value = bias_[slot * kByteClasses + byte];
                    for (std::size_t feature = 0; feature < input_dim_; ++feature) {
                        value += weights_[weight_index(slot, byte, feature)] *
                                 features[feature];
                    }
                    logits[byte] = value;
                }
                const std::size_t predicted = static_cast<std::size_t>(
                    std::distance(logits.begin(),
                                  std::max_element(logits.begin(), logits.end())));
                if (predicted == target) {
                    ++correct;
                } else {
                    example_exact = false;
                    const float margin =
                        1.0F + logits[predicted] - logits[target];
                    if (margin > 0.0F) {
                        loss += margin;
                        bias_[slot * kByteClasses + target] += learning_rate_;
                        bias_[slot * kByteClasses + predicted] -= learning_rate_;
                        for (std::size_t feature = 0; feature < input_dim_; ++feature) {
                            const float update = learning_rate_ * features[feature];
                            weights_[weight_index(slot, target, feature)] += update;
                            weights_[weight_index(slot, predicted, feature)] -= update;
                        }
                    }
                }
                ++total;
            }
            if (example_exact) {
                ++exact;
            }
        }
        std::size_t supervised_slots = 0;
        for (const DialogueExample& example : dataset) {
            supervised_slots +=
                std::min(max_response_bytes_, example.response.size() + 1);
        }
        // The response learner uses a multiclass margin objective. Keep the
        // field name for report compatibility with the initial API.
        report.cross_entropy =
            loss / static_cast<float>(std::max<std::size_t>(1, supervised_slots));
        report.byte_accuracy = static_cast<float>(correct) / static_cast<float>(total);
        report.exact_match = static_cast<float>(exact) / static_cast<float>(dataset.size());
    }
    trained_ = true;
    return report;
}

std::string HolisticDialogueModel::respond(const std::string& input,
                                            float* confidence) const {
    if (!trained_) {
        if (confidence != nullptr) {
            *confidence = 0.0F;
        }
        return {};
    }
    const math::Vector features = encode_input(input);
    std::string result;
    result.reserve(max_response_bytes_);
    float confidence_sum = 0.0F;
    std::size_t decoded_slots = 0;
    for (std::size_t slot = 0; slot < max_response_bytes_; ++slot) {
        std::size_t best = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        float second_logit = -std::numeric_limits<float>::infinity();
        for (std::size_t byte = 0; byte < kByteClasses; ++byte) {
            float value = bias_[slot * kByteClasses + byte];
            for (std::size_t feature = 0; feature < input_dim_; ++feature) {
                value += weights_[weight_index(slot, byte, feature)] * features[feature];
            }
            if (value > best_logit) {
                second_logit = best_logit;
                best_logit = value;
                best = byte;
            } else if (value > second_logit) {
                second_logit = value;
            }
        }
        if (best == 0) {
            break;
        }
        result.push_back(static_cast<char>(best));
        ++decoded_slots;
        confidence_sum +=
            1.0F / (1.0F + std::exp(-(best_logit - second_logit)));
    }
    if (confidence != nullptr) {
        *confidence = decoded_slots == 0
            ? 0.0F
            : confidence_sum / static_cast<float>(decoded_slots);
    }
    return result;
}

void HolisticDialogueModel::save(std::ostream& output) const {
    write_size(output, input_dim_);
    write_size(output, max_response_bytes_);
    output.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
    const std::uint8_t trained = trained_ ? 1U : 0U;
    output.write(reinterpret_cast<const char*>(&trained), sizeof(trained));
    write_vector(output, weights_);
    write_vector(output, bias_);
    if (!output) {
        throw std::runtime_error("failed to write dialogue checkpoint");
    }
}

void HolisticDialogueModel::load(std::istream& input) {
    if (read_size(input) != input_dim_ ||
        read_size(input) != max_response_bytes_) {
        throw std::runtime_error("dialogue checkpoint configuration mismatch");
    }
    float learning_rate = 0.0F;
    input.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    std::uint8_t trained = 0;
    input.read(reinterpret_cast<char*>(&trained), sizeof(trained));
    if (!input || !std::isfinite(learning_rate) ||
        std::abs(learning_rate - learning_rate_) > 1.0e-6F ||
        trained > 1U) {
        throw std::runtime_error("invalid dialogue checkpoint header");
    }
    weights_ = read_vector(input, weights_.size());
    bias_ = read_vector(input, bias_.size());
    trained_ = trained != 0U;
}

void HolisticDialogueModel::validate_dataset(
    const std::vector<DialogueExample>& dataset,
    std::size_t max_response_bytes) {
    if (dataset.empty() || max_response_bytes == 0) {
        throw std::invalid_argument("dialogue dataset must not be empty");
    }
    for (std::size_t index = 0; index < dataset.size(); ++index) {
        const DialogueExample& example = dataset[index];
        if (example.input.empty() || example.response.empty()) {
            throw std::invalid_argument("dialogue example " + std::to_string(index) +
                                        " must contain input and response");
        }
        if (example.response.size() > max_response_bytes ||
            example.input.find('\0') != std::string::npos ||
            example.response.find('\0') != std::string::npos ||
            example.input.find('\t') != std::string::npos ||
            example.response.find('\t') != std::string::npos) {
            throw std::invalid_argument("dialogue example " + std::to_string(index) +
                                        " contains an unsupported byte");
        }
    }
}

void HolisticDialogueModel::save_tsv(
    const std::string& path, const std::vector<DialogueExample>& dataset) {
    validate_dataset(dataset, 1000000);
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create dialogue dataset: " + path);
    }
    output << "# Astrax holistic dialogue pairs, UTF-8 text\n";
    output << "# input<TAB>response\n";
    for (const DialogueExample& example : dataset) {
        output << example.input << '\t' << example.response << '\n';
    }
}

std::vector<DialogueExample> HolisticDialogueModel::load_tsv(
    const std::string& path, std::size_t max_response_bytes) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open dialogue dataset: " + path);
    }
    std::vector<DialogueExample> dataset;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::vector<std::string> fields = split_tab(line);
        if (fields.size() != 2) {
            throw std::runtime_error("invalid dialogue TSV row");
        }
        dataset.push_back({fields[0], fields[1]});
    }
    validate_dataset(dataset, max_response_bytes);
    return dataset;
}

} // namespace astrax
