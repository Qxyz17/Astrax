#include <iostream>
#include <string>

#include "astrax/model.hpp"

int main(int argc, char** argv) {
    const std::string request = argc > 1 ? argv[1] : "observe and summarize the current goal";

    astrax::AstraxModel model;
    model.set_goal({"bootstrap", "Build a stable local digital model", 0.9F, true});
    model.remember("Osten is the fixed-size local decision engine.", 0.9F);
    model.remember("Astrax owns identity, goals, memory, introspection, and training.", 0.8F);

    const auto dataset = astrax::make_starter_dataset(
        model.config().state_dim, model.config().action_count);
    const astrax::TrainingReport report = model.train_offline(dataset, 4);
    std::cout << "trained samples=" << report.samples
              << " epochs=" << report.epochs
              << " predictor_loss=" << report.predictor_loss
              << " value_loss=" << report.value_loss
              << " intrinsic_reward=" << report.average_intrinsic_reward << '\n';

    astrax::MultimodalInput input;
    input.modality = astrax::Modality::Text;
    input.text = request;
    const astrax::ModelOutput text = model.step(input, astrax::OutputMode::Text);
    std::cout << "\n[decision]\n" << text.text << '\n';

    const astrax::ModelOutput code = model.step(input, astrax::OutputMode::Code);
    std::cout << "\n[second decision]\n" << code.text;

    const astrax::ModelOutput state = model.step(input, astrax::OutputMode::State);
    std::cout << "\n[state]\n" << state.text << '\n';
    std::cout << "\n[introspection] iteration="
              << model.introspect("How many iterations have elapsed?")
              << " goal=\"" << model.introspect("What is the current goal?")
              << "\" memories=" << model.introspect("How many memories do you have?")
              << '\n';
    return 0;
}

