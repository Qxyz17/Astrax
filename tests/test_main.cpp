#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "astrax/model.hpp"
#include "astrax/architecture.hpp"
#include "astrax/training.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_memory_and_introspection() {
    astrax::AstraxModel model;
    model.set_goal({"test", "Run the test model", 0.7F, true});
    model.remember("first memory", 0.8F);
    model.remember("second memory", 0.6F);
    require(model.recall("first", 1).size() == 1, "memory recall");

    astrax::MultimodalInput input;
    input.text = "hello";
    const auto output = model.step(input);
    require(output.text.find("decision action=") != std::string::npos,
            "text mode must render the completed decision");
    require(model.state().iteration == 1, "iteration state");
    require(model.introspect("iteration") == "1", "structured introspection");
    require(model.introspect("How many memories do you have?") == "3",
            "memory count introspection");
    require(model.introspect("What do you remember?").find("first memory") !=
                std::string::npos,
            "memory content introspection");
    require(model.introspect("我记住了什么？").find("first memory") !=
                std::string::npos,
            "Chinese memory content introspection");
}

void test_training_changes_predictor() {
    astrax::ModelConfig config;
    config.state_dim = 4;
    config.action_count = 2;
    config.learning_rate = 0.03F;
    astrax::StatePredictor predictor(config.state_dim, config.action_count, config.learning_rate);
    astrax::ActionValueModel values(config.state_dim, config.action_count, config.learning_rate);
    astrax::IntrinsicRewardModel intrinsic(config.intrinsic_reward_scale);
    astrax::OfflineRLTrainer trainer(config, predictor, values, intrinsic);
    const auto dataset = astrax::make_starter_dataset(4, 2, 2, 5);
    const auto before = predictor.predict(dataset[0].state, dataset[0].action);
    const auto report = trainer.train(dataset, 3);
    const auto after = predictor.predict(dataset[0].state, dataset[0].action);
    require(report.samples == dataset.size(), "training sample count");
    require(report.epochs == 3, "training epoch count");
    require(before != after, "predictor must learn from data");
}

void test_value_guided_action_selection() {
    astrax::ModelConfig config;
    config.state_dim = 4;
    config.action_count = 2;
    config.learning_rate = 0.2F;
    config.policy_guidance_weight = 0.25F;
    config.value_guidance_weight = 1.0F;
    astrax::AstraxModel model(config);

    osten::Decision candidate;
    candidate.action_id = 0;
    candidate.action_logits = {3.0F, 0.0F};
    candidate.state.assign(config.state_dim, 0.0F);
    require(model.select_action(candidate) == 0,
            "untrained value path must preserve the Osten candidate");

    astrax::OfflineTransition transition;
    transition.state.assign(config.state_dim, 0.0F);
    transition.next_state.assign(config.state_dim, 0.0F);
    transition.action = 1;
    transition.reward = 2.0F;
    transition.terminal = true;
    model.train_offline({transition}, 8);
    require(model.values().value(transition.state, 1) >
                model.values().value(transition.state, 0),
            "offline value training must separate action values");
    require(model.select_action(candidate) == 1,
            "learned value path must change final action selection");
}

void test_online_feedback_updates_value_path() {
    astrax::ModelConfig config;
    config.state_dim = 4;
    config.action_count = 2;
    config.learning_rate = 0.2F;
    config.policy_guidance_weight = 0.0F;
    config.value_guidance_weight = 1.0F;
    astrax::AstraxModel model(config);
    astrax::MultimodalInput input;
    input.text = "feedback transition";
    const auto output = model.step(input);
    const float before = model.values().value(output.state, output.action_id);
    model.observe_feedback(output.state, 2.0F, true);
    const float after = model.values().value(output.state, output.action_id);
    require(after > before, "online feedback must update the selected value");
}

void test_checkpoint_round_trip() {
    astrax::ModelConfig config;
    config.state_dim = 4;
    config.action_count = 2;
    astrax::AstraxModel trained(config);
    const auto dataset = astrax::make_starter_dataset(4, 2, 2, 5);
    trained.train_offline(dataset, 3);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "astrax-test-checkpoint.astrax-model";
    trained.save_checkpoint(path.string());
    astrax::AstraxModel restored(config);
    restored.load_checkpoint(path.string());
    std::filesystem::remove(path);

    require(trained.predictor().predict(dataset[0].state, dataset[0].action) ==
                restored.predictor().predict(dataset[0].state, dataset[0].action),
            "checkpoint predictor round trip");
    require(trained.values().value(dataset[0].state, dataset[0].action) ==
                restored.values().value(dataset[0].state, dataset[0].action),
            "checkpoint value round trip");
}

void test_architecture_contract() {
    static_assert(!astrax::architecture::kUsesTransformer);
    static_assert(!astrax::architecture::kUsesNextTokenPrediction);
    static_assert(!astrax::architecture::kUsesAutoregressiveGeneration);
    static_assert(astrax::architecture::kOstenIsCoreDecisionEngine);
    static_assert(astrax::architecture::kUsesParallelResponseSlots);
    require(!astrax::architecture::kFinalDecisionUsesKeywordRules,
            "keyword rules must not select final answers");
    require(!astrax::architecture::kFinalDecisionUsesDirectRetrieval,
            "retrieval must not select final answers");
}

void test_holistic_dialogue() {
    astrax::HolisticDialogueModel dialogue(96, 32, 0.12F);
    const std::vector<astrax::TextDocument> dataset = {
        {"alpha question alpha response"},
        {"beta question beta response"},
        {"gamma question gamma response"}
    };
    const auto report = dialogue.train(dataset, 8);
    require(report.examples == dataset.size(), "dialogue sample count");
    require(dialogue.trained(), "dialogue must be marked trained");
    const std::string response = dialogue.respond("novel input");
    require(!response.empty(), "dialogue must decode a learned Unicode response");
    require(response.find_first_not_of(" \t\r\n") != std::string::npos,
            "dialogue must not decode whitespace-only output");
    require(response.find(u8"\uFFFD") == std::string::npos,
            "dialogue must not emit replacement characters");
    require(!dialogue.respond(u8"中文输入和 English input").empty(),
            "dialogue must accept mixed UTF-8 input");
}

void test_dialogue_pairs() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "astrax-dialogue-pairs.tsv";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << "hello\tHello, Astrax.\n"
               << "write code\tint main() { return 0; }\n";
    }
    const auto pairs = astrax::HolisticDialogueModel::load_pairs(path.string(), 4096);
    std::filesystem::remove(path);
    require(pairs.size() == 2, "dialogue pair count");
    require(pairs[0].input == "hello" && pairs[0].target == "Hello, Astrax.",
            "dialogue pair fields");
    astrax::HolisticDialogueModel dialogue(96, 32, 0.05F);
    const auto report = dialogue.train_pairs(pairs, 4);
    require(report.examples == pairs.size(), "dialogue pair training count");
    require(dialogue.trained(), "dialogue pair model must be trained");
}

} // namespace

int main() {
    try {
        test_architecture_contract();
        test_memory_and_introspection();
        test_training_changes_predictor();
        test_checkpoint_round_trip();
        test_value_guided_action_selection();
        test_online_feedback_updates_value_path();
        test_holistic_dialogue();
        test_dialogue_pairs();
        std::cout << "all Astrax tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
