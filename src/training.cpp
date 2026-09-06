#include "astrax/training.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
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
        result.push_back(std::stof(field));
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
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        float predictor_loss = 0.0F;
        float value_loss = 0.0F;
        float intrinsic = 0.0F;
        for (const OfflineTransition& sample : dataset) {
            const math::Vector predicted_next = predictor_.predict(sample.state, sample.action);
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
        if (fields.size() != 5 || fields[0] == "action") {
            if (fields.size() == 5 && fields[0] == "action") {
                continue;
            }
            throw std::runtime_error("invalid offline dataset row " +
                                     std::to_string(line_number));
        }
        OfflineTransition sample;
        sample.action = static_cast<std::size_t>(std::stoul(fields[0]));
        sample.reward = std::stof(fields[1]);
        sample.terminal = std::stoi(fields[2]) != 0;
        sample.state = parse_vector(fields[3], state_dim);
        sample.next_state = parse_vector(fields[4], state_dim);
        dataset.push_back(std::move(sample));
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
    std::uniform_real_distribution<float> noise(-0.03F, 0.03F);
    std::vector<OfflineTransition> dataset;
    dataset.reserve(episodes * horizon);
    for (std::size_t episode = 0; episode < episodes; ++episode) {
        math::Vector state(state_dim, 0.0F);
        for (std::size_t step = 0; step < horizon; ++step) {
            const std::size_t action = (episode + step) % action_count;
            math::Vector next_state = state;
            const std::size_t coordinate = action % state_dim;
            next_state[coordinate] = std::clamp(
                next_state[coordinate] + 0.08F + noise(rng), -1.0F, 1.0F);
            for (std::size_t index = 0; index < state_dim; ++index) {
                if (index != coordinate) {
                    next_state[index] = std::clamp(
                        next_state[index] * 0.995F + noise(rng) * 0.2F, -1.0F, 1.0F);
                }
            }
            OfflineTransition sample;
            sample.state = state;
            sample.action = action;
            sample.reward = 1.0F - std::abs(0.5F - next_state[coordinate]);
            sample.next_state = next_state;
            sample.terminal = step + 1 == horizon;
            dataset.push_back(sample);
            state = std::move(next_state);
        }
    }
    return dataset;
}

} // namespace astrax
