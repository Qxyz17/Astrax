#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "astrax/model.hpp"

namespace {

std::filesystem::path project_root() {
    std::filesystem::path root = std::filesystem::current_path();
    if (std::filesystem::exists(root / "todolist.md")) {
        return root;
    }
    if (root.filename() == "Release" || root.filename() == "Debug") {
        root = root.parent_path().parent_path();
    }
    return root;
}

} // namespace

int main() {
    try {
        astrax::ModelConfig config;
        config.state_dim = 32;
        config.action_count = 8;
        const std::size_t episodes = 128;
        const std::size_t horizon = 32;
        const std::size_t epochs = 24;
        const std::size_t dialogue_epochs = 160;

        const std::filesystem::path root = project_root();
        const std::filesystem::path data_directory = root / "data";
        const std::filesystem::path artifact_directory = root / "artifacts";
        std::filesystem::create_directories(data_directory);
        std::filesystem::create_directories(artifact_directory);

        const std::filesystem::path dataset_path =
            data_directory / "astrax_offline_transitions.csv";
        const auto generated = astrax::make_starter_dataset(
            config.state_dim, config.action_count, episodes, horizon);
        astrax::OfflineRLTrainer::validate_dataset(
            generated, config.state_dim, config.action_count);
        astrax::OfflineRLTrainer::save_csv(dataset_path.string(), generated);

        const auto loaded = astrax::OfflineRLTrainer::load_csv(
            dataset_path.string(), config.state_dim);
        astrax::AstraxModel model(config);
        const astrax::TrainingReport report =
            model.train_offline(loaded, epochs);
        const std::filesystem::path dialogue_dataset_path =
            data_directory / "astrax_dialogue_pairs.tsv";
        const std::vector<astrax::DialogueExample> dialogue_dataset =
            astrax::HolisticDialogueModel::load_tsv(
                dialogue_dataset_path.string(),
                model.dialogue().max_response_bytes());
        const astrax::DialogueTrainingReport dialogue_report =
            model.train_dialogue(dialogue_dataset, dialogue_epochs);

        const std::filesystem::path checkpoint_path =
            artifact_directory / "astrax_training.astrax-model";
        const astrax::math::Vector validation_state = loaded.front().state;
        const std::size_t validation_action = loaded.front().action;
        const astrax::math::Vector expected_prediction =
            model.predictor().predict(validation_state, validation_action);
        const float expected_value =
            model.values().value(validation_state, validation_action);
        const std::string expected_dialogue =
            model.dialogue().respond(dialogue_dataset.front().input);
        model.save_checkpoint(checkpoint_path.string());

        astrax::AstraxModel restored_model(config);
        restored_model.load_checkpoint(checkpoint_path.string());
        const astrax::math::Vector restored_prediction =
            restored_model.predictor().predict(validation_state, validation_action);
        const float restored_value =
            restored_model.values().value(validation_state, validation_action);
        const std::string restored_dialogue =
            restored_model.dialogue().respond(dialogue_dataset.front().input);
        if (expected_prediction != restored_prediction ||
            expected_value != restored_value ||
            expected_dialogue != restored_dialogue ||
            !restored_model.dialogue().trained()) {
            throw std::runtime_error("checkpoint reload verification failed");
        }

        const std::filesystem::path report_path =
            artifact_directory / "offline_training_report.txt";
        std::ofstream report_file(report_path, std::ios::trunc);
        report_file << std::setprecision(9)
                    << "dataset=" << dataset_path.string() << '\n'
                    << "samples=" << report.samples << '\n'
                    << "epochs=" << report.epochs << '\n'
                    << "predictor_loss=" << report.predictor_loss << '\n'
                    << "value_loss=" << report.value_loss << '\n'
                    << "average_intrinsic_reward="
                    << report.average_intrinsic_reward << '\n'
                    << "dialogue_dataset=" << dialogue_dataset_path.string() << '\n'
                    << "dialogue_examples=" << dialogue_report.examples << '\n'
                    << "dialogue_epochs=" << dialogue_report.epochs << '\n'
                    << "dialogue_cross_entropy="
                    << dialogue_report.cross_entropy << '\n'
                    << "dialogue_byte_accuracy="
                    << dialogue_report.byte_accuracy << '\n'
                    << "dialogue_exact_match="
                    << dialogue_report.exact_match << '\n'
                    << "checkpoint=" << checkpoint_path.string() << '\n'
                    << "checkpoint_verified=1\n";

        std::cout << "offline training completed\n"
                  << "dataset=" << dataset_path.string() << '\n'
                  << "samples=" << report.samples << '\n'
                  << "episodes=" << episodes << '\n'
                  << "horizon=" << horizon << '\n'
                  << "epochs=" << report.epochs << '\n'
                  << "predictor_loss=" << report.predictor_loss << '\n'
                  << "value_loss=" << report.value_loss << '\n'
                  << "average_intrinsic_reward="
                  << report.average_intrinsic_reward << '\n'
                  << "dialogue_examples=" << dialogue_report.examples << '\n'
                  << "dialogue_epochs=" << dialogue_report.epochs << '\n'
                  << "dialogue_cross_entropy="
                  << dialogue_report.cross_entropy << '\n'
                  << "dialogue_byte_accuracy="
                  << dialogue_report.byte_accuracy << '\n'
                  << "dialogue_exact_match="
                  << dialogue_report.exact_match << '\n'
                  << "checkpoint=" << checkpoint_path.string() << '\n'
                  << "checkpoint_verified=1\n"
                  << "report=" << report_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "training failure: " << error.what() << '\n';
        return 1;
    }
}
