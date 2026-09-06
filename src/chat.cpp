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

std::string utf8_from_wide(const std::wstring& value) {
#ifdef _WIN32
    if (value.empty()) {
        return {};
    }
    const int size = WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        throw std::runtime_error("cannot convert Windows text to UTF-8");
    }
    std::string result(static_cast<std::size_t>(size), '\0');
    if (WideCharToMultiByte(
            CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
            static_cast<int>(value.size()), result.data(), size, nullptr, nullptr) <= 0) {
        throw std::runtime_error("cannot convert Windows text to UTF-8");
    }
    return result;
#else
    return std::string(value.begin(), value.end());
#endif
}

} // namespace

int wmain(int argc, wchar_t** argv) {
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
            std::string message = utf8_from_wide(argv[1]);
            for (int index = 2; index < argc; ++index) {
                message += " ";
                message += utf8_from_wide(argv[index]);
            }
            std::cout << model.chat(message) << '\n';
            return 0;
        }

        std::cout << "Astrax dialogue ready. 输入 /exit 结束。\n";
        std::wstring wide_message;
        while (true) {
            std::cout << "你> " << std::flush;
            if (!std::getline(std::wcin, wide_message) ||
                wide_message == L"/exit") {
                break;
            }
            const std::string message = utf8_from_wide(wide_message);
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
