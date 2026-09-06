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
                    << report.average_intrinsic_reward << '\n';

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
                  << "report=" << report_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "training failure: " << error.what() << '\n';
        return 1;
    }
}
