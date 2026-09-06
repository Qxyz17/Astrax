#include "astrax/training.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <istream>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "astrax/math.hpp"

namespace astrax {

namespace {

math::Vector features_for(const math::Vector& state,
                          std::size_t action,
                          std::size_t action_count) {
    math::Vector features = state;
    const math::Vector one_hot = math::one_hot(action, action_count);
    features.insert(features.end(), one_hot.begin(), one_hot.end());
    return features;
}

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> fields;
    std::stringstream stream(value);
    std::string field;
    while (std::getline(stream, field, separator)) {
        fields.push_back(field);
    }
    return fields;
}

math::Vector parse_vector(const std::string& value, std::size_t expected) {
    const std::vector<std::string> fields = split(value, ';');
    if (fields.size() != expected) {
        throw std::runtime_error("offline dataset vector has an unexpected dimension");
    }
    math::Vector result;
    result.reserve(expected);
    for (const std::string& field : fields) {
        const float item = std::stof(field);
        if (!std::isfinite(item)) {
            throw std::runtime_error("offline dataset contains a non-finite vector value");
        }
        result.push_back(item);
    }
    return result;
}

void validate_state(const math::Vector& state, std::size_t state_dim,
                    const char* name) {
    math::require_size(state, state_dim, name);
    for (float item : state) {
        if (!std::isfinite(item) || item < -1.0F || item > 1.0F) {
            throw std::invalid_argument(std::string(name) +
                                        " must contain finite values in [-1, 1]");
        }
    }
}

void write_size(std::ostream& output, std::size_t value) {
    const std::uint64_t stored = static_cast<std::uint64_t>(value);
    output.write(reinterpret_cast<const char*>(&stored), sizeof(stored));
}

std::size_t read_size(std::istream& input) {
    std::uint64_t stored = 0;
    input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
    if (!input) {
        throw std::runtime_error("checkpoint ended while reading a size");
    }
    return static_cast<std::size_t>(stored);
}

void write_vector(std::ostream& output, const math::Vector& value) {
    write_size(output, value.size());
    output.write(reinterpret_cast<const char*>(value.data()),
                 static_cast<std::streamsize>(value.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write checkpoint vector");
    }
}

math::Vector read_vector(std::istream& input, std::size_t expected_size) {
    const std::size_t stored_size = read_size(input);
    if (stored_size != expected_size) {
        throw std::runtime_error("checkpoint vector dimension does not match model");
    }
    math::Vector result(stored_size, 0.0F);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("checkpoint ended while reading a vector");
    }
    for (float item : result) {
        if (!std::isfinite(item)) {
            throw std::runtime_error("checkpoint contains a non-finite weight");
        }
    }
    return result;
}

math::Vector read_vector(std::istream& input) {
    const std::size_t stored_size = read_size(input);
    if (stored_size > 100000000ULL) {
        throw std::runtime_error("checkpoint vector is unreasonably large");
    }
    math::Vector result(stored_size, 0.0F);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("checkpoint ended while reading a vector");
    }
    for (float item : result) {
        if (!std::isfinite(item)) {
            throw std::runtime_error("checkpoint contains a non-finite value");
        }
    }
    return result;
}

} // namespace

StatePredictor::StatePredictor(std::size_t state_dim,
                               std::size_t action_count,
                               float learning_rate)
    : state_dim_(state_dim),
      action_count_(action_count),
      learning_rate_(learning_rate),
      weights_(state_dim * (state_dim + action_count), 0.0F),
      bias_(state_dim, 0.0F) {}

math::Vector StatePredictor::predict(const math::Vector& state,
                                     std::size_t action) const {
    math::require_size(state, state_dim_, "predictor state");
    const math::Vector features = features_for(state, action, action_count_);
    math::Vector result(state_dim_, 0.0F);
    const std::size_t input_dim = features.size();
    for (std::size_t row = 0; row < state_dim_; ++row) {
        result[row] = bias_[row];
        for (std::size_t column = 0; column < input_dim; ++column) {
            result[row] += weights_[row * input_dim + column] * features[column];
        }
    }
    return result;
}

float StatePredictor::train(const OfflineTransition& sample) {
    math::require_size(sample.state, state_dim_, "predictor sample state");
    math::require_size(sample.next_state, state_dim_, "predictor sample next state");
    const math::Vector features = features_for(sample.state, sample.action, action_count_);
    const math::Vector prediction = predict(sample.state, sample.action);
    const std::size_t input_dim = features.size();
    float loss = 0.0F;
    for (std::size_t row = 0; row < state_dim_; ++row) {
        const float error = sample.next_state[row] - prediction[row];
        loss += error * error;
        for (std::size_t column = 0; column < input_dim; ++column) {
            weights_[row * input_dim + column] +=
                learning_rate_ * error * features[column];
        }
        bias_[row] += learning_rate_ * error;
    }
    return loss / static_cast<float>(state_dim_);
}

void StatePredictor::save(std::ostream& output) const {
    write_size(output, state_dim_);
    write_size(output, action_count_);
    write_vector(output, weights_);
    write_vector(output, bias_);
}

void StatePredictor::load(std::istream& input) {
    if (read_size(input) != state_dim_ || read_size(input) != action_count_) {
        throw std::runtime_error("checkpoint predictor dimensions do not match model");
    }
    weights_ = read_vector(input, weights_.size());
    bias_ = read_vector(input, bias_.size());
}

ActionValueModel::ActionValueModel(std::size_t state_dim,
                                   std::size_t action_count,
                                   float learning_rate)
    : state_dim_(state_dim),
      action_count_(action_count),
      learning_rate_(learning_rate),
      weights_(state_dim * action_count, 0.0F),
      bias_(action_count, 0.0F) {}

float ActionValueModel::value(const math::Vector& state, std::size_t action) const {
    math::require_size(state, state_dim_, "value state");
    if (action >= action_count_) {
        throw std::invalid_argument("value action is out of range");
    }
    float result = bias_[action];
    for (std::size_t index = 0; index < state_dim_; ++index) {
        result += weights_[action * state_dim_ + index] * state[index];
    }
    return result;
}

math::Vector ActionValueModel::values(const math::Vector& state) const {
    math::Vector result(action_count_, 0.0F);
    for (std::size_t action = 0; action < action_count_; ++action) {
        result[action] = value(state, action);
    }
    return result;
}

float ActionValueModel::train_q_learning(const OfflineTransition& sample,
                                         float discount,
                                         const math::Vector& next_state) {
    const float current = value(sample.state, sample.action);
    const math::Vector next_values = values(next_state);
    const float next = sample.terminal
        ? 0.0F
        : *std::max_element(next_values.begin(), next_values.end());
    const float target = sample.reward + discount * next;
    (void)current;
    return train_supervised(sample, target);
}

float ActionValueModel::train_supervised(const OfflineTransition& sample, float target) {
    math::require_size(sample.state, state_dim_, "value sample state");
    if (sample.action >= action_count_) {
        throw std::invalid_argument("value sample action is out of range");
    }
    const float error = target - value(sample.state, sample.action);
    for (std::size_t index = 0; index < state_dim_; ++index) {
        weights_[sample.action * state_dim_ + index] +=
            learning_rate_ * error * sample.state[index];
    }
    bias_[sample.action] += learning_rate_ * error;
    return error * error;
}

void ActionValueModel::save(std::ostream& output) const {
    write_size(output, state_dim_);
    write_size(output, action_count_);
    write_vector(output, weights_);
    write_vector(output, bias_);
}

void ActionValueModel::load(std::istream& input) {
    if (read_size(input) != state_dim_ || read_size(input) != action_count_) {
        throw std::runtime_error("checkpoint value dimensions do not match model");
    }
    weights_ = read_vector(input, weights_.size());
    bias_ = read_vector(input, bias_.size());
}

IntrinsicRewardModel::IntrinsicRewardModel(float scale) : scale_(scale) {}

float IntrinsicRewardModel::observe(const math::Vector& state) {
    if (observations_ == 0) {
        centroid_ = state;
        observations_ = 1;
        return scale_;
    }
    math::require_size(state, centroid_.size(), "intrinsic state");
    const float novelty = std::min(1.0F, math::norm(math::subtract(state, centroid_)));
    const float count = static_cast<float>(observations_);
    for (std::size_t index = 0; index < centroid_.size(); ++index) {
        centroid_[index] = (centroid_[index] * count + state[index]) / (count + 1.0F);
    }
    ++observations_;
    return scale_ * novelty;
}

void IntrinsicRewardModel::clear() noexcept {
    observations_ = 0;
    centroid_.clear();
}

void IntrinsicRewardModel::save(std::ostream& output) const {
    output.write(reinterpret_cast<const char*>(&scale_), sizeof(scale_));
    write_size(output, observations_);
    write_vector(output, centroid_);
    if (!output) {
        throw std::runtime_error("failed to write intrinsic reward checkpoint");
    }
}

void IntrinsicRewardModel::load(std::istream& input) {
    float stored_scale = 0.0F;
    input.read(reinterpret_cast<char*>(&stored_scale), sizeof(stored_scale));
    if (!input || !std::isfinite(stored_scale) ||
        std::abs(stored_scale - scale_) > 1.0e-6F) {
        throw std::runtime_error("checkpoint intrinsic reward configuration mismatch");
    }
    const std::size_t stored_observations = read_size(input);
    math::Vector stored_centroid = read_vector(input);
    if ((stored_observations == 0 && !stored_centroid.empty()) ||
        (stored_observations != 0 && stored_centroid.empty())) {
        throw std::runtime_error("checkpoint intrinsic reward state is inconsistent");
    }
    centroid_ = std::move(stored_centroid);
    observations_ = stored_observations;
}
OfflineRLTrainer::OfflineRLTrainer(const ModelConfig& config,
                                   StatePredictor& predictor,
                                   ActionValueModel& values,
                                   IntrinsicRewardModel& intrinsic)
    : config_(config), predictor_(predictor), values_(values), intrinsic_(intrinsic) {}

TrainingReport OfflineRLTrainer::train(const std::vector<OfflineTransition>& dataset,
                                       std::size_t epochs) {
    TrainingReport report;
    report.samples = dataset.size();
    report.epochs = epochs;
    if (dataset.empty() || epochs == 0) {
        return report;
    }
    validate_dataset(dataset, config_.state_dim, config_.action_count);
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        float predictor_loss = 0.0F;
        float value_loss = 0.0F;
        float intrinsic = 0.0F;
        for (const OfflineTransition& sample : dataset) {
            predictor_loss += predictor_.train(sample);
            value_loss += values_.train_q_learning(sample, config_.discount,
                                                   sample.next_state);
            intrinsic += intrinsic_.observe(sample.next_state);
        }
        report.predictor_loss = predictor_loss / static_cast<float>(dataset.size());
        report.value_loss = value_loss / static_cast<float>(dataset.size());
        report.average_intrinsic_reward = intrinsic / static_cast<float>(dataset.size());
    }
    return report;
}

void OfflineRLTrainer::validate_dataset(const std::vector<OfflineTransition>& dataset,
                                        std::size_t state_dim,
                                        std::size_t action_count) {
    if (state_dim == 0 || action_count == 0) {
        throw std::invalid_argument("offline dataset dimensions must be positive");
    }
    if (dataset.empty()) {
        throw std::invalid_argument("offline dataset must not be empty");
    }
    for (std::size_t index = 0; index < dataset.size(); ++index) {
        const OfflineTransition& sample = dataset[index];
        try {
            validate_state(sample.state, state_dim, "offline state");
            validate_state(sample.next_state, state_dim, "offline next_state");
        } catch (const std::exception& error) {
            throw std::invalid_argument(
                "invalid offline dataset sample " + std::to_string(index) +
                ": " + error.what());
        }
        if (sample.action >= action_count) {
            throw std::invalid_argument(
                "invalid offline dataset sample " + std::to_string(index) +
                ": action is out of range");
        }
        if (!std::isfinite(sample.reward)) {
            throw std::invalid_argument(
                "invalid offline dataset sample " + std::to_string(index) +
                ": reward is not finite");
        }
    }
}

void OfflineRLTrainer::save_csv(const std::string& path,
                                const std::vector<OfflineTransition>& dataset) {
    if (dataset.empty()) {
        throw std::invalid_argument("cannot save an empty offline dataset");
    }
    const std::size_t state_dim = dataset.front().state.size();
    std::size_t action_count = 0;
    for (const OfflineTransition& sample : dataset) {
        action_count = std::max(action_count, sample.action + 1);
    }
    validate_dataset(dataset, state_dim, action_count);

    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create offline dataset: " + path);
    }
    output << std::setprecision(9);
    output << "# Astrax offline transition dataset\n";
    output << "# state values are finite and bounded to [-1, 1]\n";
    output << "action,reward,terminal,state,next_state\n";
    for (const OfflineTransition& sample : dataset) {
        output << sample.action << ',' << sample.reward << ','
               << (sample.terminal ? 1 : 0) << ',';
        for (std::size_t index = 0; index < sample.state.size(); ++index) {
            if (index != 0) {
                output << ';';
            }
            output << sample.state[index];
        }
        output << ',';
        for (std::size_t index = 0; index < sample.next_state.size(); ++index) {
            if (index != 0) {
                output << ';';
            }
            output << sample.next_state[index];
        }
        output << '\n';
    }
}

std::vector<OfflineTransition> OfflineRLTrainer::load_csv(const std::string& path,
                                                           std::size_t state_dim) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open offline dataset: " + path);
    }
    std::vector<OfflineTransition> dataset;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::vector<std::string> fields = split(line, ',');
        if (fields.size() == 5 && fields[0] == "action") {
            continue;
        }
        if (fields.size() != 5) {
            throw std::runtime_error("invalid offline dataset row " +
                                     std::to_string(line_number));
        }
        try {
            OfflineTransition sample;
            sample.action = static_cast<std::size_t>(std::stoul(fields[0]));
            sample.reward = std::stof(fields[1]);
            const int terminal = std::stoi(fields[2]);
            if (terminal != 0 && terminal != 1) {
                throw std::runtime_error("terminal must be 0 or 1");
            }
            sample.terminal = terminal != 0;
            sample.state = parse_vector(fields[3], state_dim);
            sample.next_state = parse_vector(fields[4], state_dim);
            dataset.push_back(std::move(sample));
        } catch (const std::exception& error) {
            throw std::runtime_error("invalid offline dataset row " +
                                     std::to_string(line_number) + ": " +
                                     error.what());
        }
    }
    if (!dataset.empty()) {
        std::size_t action_count = 0;
        for (const OfflineTransition& sample : dataset) {
            action_count = std::max(action_count, sample.action + 1);
        }
        validate_dataset(dataset, state_dim, action_count);
    }
    return dataset;
}

std::vector<OfflineTransition> make_starter_dataset(std::size_t state_dim,
                                                     std::size_t action_count,
                                                     std::size_t episodes,
                                                     std::size_t horizon) {
    if (state_dim == 0 || action_count == 0) {
        throw std::invalid_argument("starter dataset dimensions must be positive");
    }
    std::mt19937 rng(91);
    std::uniform_real_distribution<float> noise(-0.01F, 0.01F);
    std::uniform_real_distribution<float> target_distribution(-0.8F, 0.8F);
    std::vector<OfflineTransition> dataset;
    dataset.reserve(episodes * horizon);
    math::Vector target(state_dim, 0.0F);
    for (std::size_t index = 0; index < state_dim; ++index) {
        target[index] = target_distribution(rng);
    }
    for (std::size_t episode = 0; episode < episodes; ++episode) {
        math::Vector state(state_dim, 0.0F);
        for (std::size_t step = 0; step < horizon; ++step) {
            const std::size_t action =
                (episode * 5 + step * 3 + (step / 4)) % action_count;
            math::Vector next_state = state;
            const std::size_t coordinate = action % state_dim;
            const float before_error = std::abs(target[coordinate] - state[coordinate]);
            const float direction = target[coordinate] >= state[coordinate] ? 1.0F : -1.0F;
            next_state[coordinate] = std::clamp(
                state[coordinate] + direction * 0.12F + noise(rng), -1.0F, 1.0F);
            for (std::size_t index = 0; index < state_dim; ++index) {
                if (index != coordinate) {
                    next_state[index] = std::clamp(
                        state[index] * 0.998F + noise(rng) * 0.15F, -1.0F, 1.0F);
                }
            }
            const float after_error = std::abs(target[coordinate] - next_state[coordinate]);
            OfflineTransition sample;
            sample.state = state;
            sample.action = action;
            sample.reward = std::clamp(before_error - after_error - 0.01F, -1.0F, 1.0F);
            sample.next_state = next_state;
            sample.terminal = step + 1 == horizon;
            dataset.push_back(sample);
            state = std::move(next_state);
        }
    }
    return dataset;
}

} // namespace astrax
