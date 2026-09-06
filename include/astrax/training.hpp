#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "astrax/types.hpp"

namespace astrax {

class StatePredictor {
public:
    StatePredictor(std::size_t state_dim, std::size_t action_count, float learning_rate);

    math::Vector predict(const math::Vector& state, std::size_t action) const;
    float train(const OfflineTransition& sample);
    std::size_t input_dim() const noexcept { return state_dim_ + action_count_; }

private:
    std::size_t state_dim_;
    std::size_t action_count_;
    float learning_rate_;
    math::Vector weights_;
    math::Vector bias_;
};

class ActionValueModel {
public:
    ActionValueModel(std::size_t state_dim, std::size_t action_count, float learning_rate);

    float value(const math::Vector& state, std::size_t action) const;
    math::Vector values(const math::Vector& state) const;
    float train_q_learning(const OfflineTransition& sample,
                           float discount,
                           const math::Vector& next_state);
    float train_supervised(const OfflineTransition& sample, float target);

private:
    std::size_t state_dim_;
    std::size_t action_count_;
    float learning_rate_;
    math::Vector weights_;
    math::Vector bias_;
};

class IntrinsicRewardModel {
public:
    explicit IntrinsicRewardModel(float scale);

    float observe(const math::Vector& state);
    void clear() noexcept;
    std::size_t observations() const noexcept { return observations_; }

private:
    float scale_;
    std::size_t observations_ = 0;
    math::Vector centroid_;
};

class OfflineRLTrainer {
public:
    OfflineRLTrainer(const ModelConfig& config,
                     StatePredictor& predictor,
                     ActionValueModel& values,
                     IntrinsicRewardModel& intrinsic);

    TrainingReport train(const std::vector<OfflineTransition>& dataset,
                         std::size_t epochs);
    static void validate_dataset(const std::vector<OfflineTransition>& dataset,
                                 std::size_t state_dim,
                                 std::size_t action_count);
    static void save_csv(const std::string& path,
                         const std::vector<OfflineTransition>& dataset);
    static std::vector<OfflineTransition> load_csv(const std::string& path,
                                                   std::size_t state_dim);

private:
    ModelConfig config_;
    StatePredictor& predictor_;
    ActionValueModel& values_;
    IntrinsicRewardModel& intrinsic_;
};

std::vector<OfflineTransition> make_starter_dataset(std::size_t state_dim,
                                                     std::size_t action_count,
                                                     std::size_t episodes = 8,
                                                     std::size_t horizon = 16);

} // namespace astrax
