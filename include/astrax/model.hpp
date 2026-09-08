#pragma once

#include <string>

#include "osten/engine.hpp"
#include "astrax/dialogue.hpp"
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
    float observe_feedback(float reward, bool terminal = false);
    float observe_feedback(const math::Vector& next_state,
                           float reward,
                           bool terminal = false);
    std::string introspect(const std::string& question) const;

    // Selects the final action from Osten's candidate policy and Astrax's
    // learned value model. This is the feedback path from training to action.
    std::size_t select_action(const osten::Decision& decision) const;

    TrainingReport train_offline(const std::vector<OfflineTransition>& dataset,
                                 std::size_t epochs);
    TrainingReport train_offline_csv(const std::string& path, std::size_t epochs);
    DialogueTrainingReport train_dialogue(
        const std::vector<TextDocument>& dataset, std::size_t epochs);
    DialogueTrainingReport train_dialogue_documents(const std::string& path,
                                                    std::size_t epochs);
    DialogueTrainingReport train_dialogue_pairs(const std::string& path,
                                                std::size_t epochs);
    DialogueTrainingReport train_dialogue_pairs(
        const std::vector<TextDocument>& dataset, std::size_t epochs);
    std::string chat(const std::string& input);
    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);

    const AstraxState& state() const noexcept { return state_; }
    const ModelConfig& config() const noexcept { return config_; }
    const StatePredictor& predictor() const noexcept { return predictor_; }
    const ActionValueModel& values() const noexcept { return values_; }
    const HolisticDialogueModel& dialogue() const noexcept { return dialogue_; }

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
    HolisticDialogueModel dialogue_;
    ControlledRenderer renderer_;
    math::Vector pending_state_;
    std::size_t pending_action_ = 0;
    bool has_pending_transition_ = false;
};

} // namespace astrax
