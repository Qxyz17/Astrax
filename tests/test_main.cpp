#include <cmath>
#include <iostream>
#include <stdexcept>

#include "astrax/model.hpp"
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

} // namespace

int main() {
    try {
        test_memory_and_introspection();
        test_training_changes_predictor();
        std::cout << "all Astrax tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
