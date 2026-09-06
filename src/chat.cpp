#include <filesystem>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        const std::filesystem::path root = project_root();
        const std::filesystem::path checkpoint =
            root / "artifacts" / "astrax_training.astrax-model";
        astrax::AstraxModel model;
        if (std::filesystem::exists(checkpoint)) {
            model.load_checkpoint(checkpoint.string());
        } else {
            const std::filesystem::path dataset =
                root / "data" / "astrax_dialogue_pairs.tsv";
            model.train_dialogue_tsv(dataset.string(), 160);
        }
        model.set_goal(
            {"conversation", "Understand the complete message and answer it", 0.9F, true});

        if (argc > 1) {
            std::string message = argv[1];
            for (int index = 2; index < argc; ++index) {
                message += " ";
                message += argv[index];
            }
            std::cout << model.chat(message) << '\n';
            return 0;
        }

        std::cout << "Astrax dialogue ready. 输入 /exit 结束。\n";
        std::string message;
        while (true) {
            std::cout << "你> " << std::flush;
            if (!std::getline(std::cin, message) || message == "/exit") {
                break;
            }
            if (message.empty()) {
                continue;
            }
            std::cout << "Astrax> " << model.chat(message) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "chat failure: " << error.what() << '\n';
        return 1;
    }
}
