#pragma once

#include <string>

#include "osten/engine.hpp"
#include "astrax/memory.hpp"
#include "astrax/renderer.hpp"
#include "astrax/training.hpp"
#include "astrax/types.hpp"

namespace astrax {

class AstraxModel {
public:
    explicit AstraxModel(ModelConfig config = {});

    void set_goal(Goal goal);
    void clear_goal();
    void remember(const std::string& content, float salience = 0.5F);
    std::vector<MemoryRecord> recall(const std::string& query, std::size_t limit = 4);

    ModelOutput step(const MultimodalInput& input,
                     OutputMode mode = OutputMode::Text);
    std::string introspect(const std::string& question) const;

    TrainingReport train_offline(const std::vector<OfflineTransition>& dataset,
                                 std::size_t epochs);
    TrainingReport train_offline_csv(const std::string& path, std::size_t epochs);

    const AstraxState& state() const noexcept { return state_; }
    const ModelConfig& config() const noexcept { return config_; }
    const StatePredictor& predictor() const noexcept { return predictor_; }
    const ActionValueModel& values() const noexcept { return values_; }

private:
    math::Vector embed(const MultimodalInput& input) const;
    std::string context_for(const math::Vector& query);

    ModelConfig config_;
    AstraxState state_;
    MemoryStore memory_;
    osten::OstenEngine engine_;
    StatePredictor predictor_;
    ActionValueModel values_;
    IntrinsicRewardModel intrinsic_;
    OfflineRLTrainer trainer_;
    ControlledRenderer renderer_;
};

} // namespace astrax
