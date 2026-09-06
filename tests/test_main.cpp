#include <cmath>
#include <filesystem>
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
    require(!output.text.empty(), "text output");
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
    astrax::HolisticDialogueModel dialogue(256, 128, 0.12F);
    const std::vector<astrax::DialogueExample> dataset = {
        {"alpha question", "alpha response"},
        {"beta question", "beta response"},
        {"gamma question", "gamma response"}
    };
    const auto report = dialogue.train(dataset, 80);
    require(report.examples == dataset.size(), "dialogue sample count");
    require(dialogue.trained(), "dialogue must be marked trained");
    require(dialogue.respond("alpha question") == "alpha response",
            "dialogue must decode learned complete response");
    require(dialogue.respond("beta question") == "beta response",
            "dialogue must separate complete inputs");
}

} // namespace

int main() {
    try {
        test_architecture_contract();
        test_memory_and_introspection();
        test_training_changes_predictor();
        test_checkpoint_round_trip();
        test_holistic_dialogue();
        std::cout << "all Astrax tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
