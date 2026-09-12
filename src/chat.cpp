#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "astrax/model.hpp"
#include "embedded_dialogue_pairs.hpp"
#include "embedded_model.hpp"

namespace {

std::vector<astrax::TextDocument> embedded_pairs() {
    std::vector<astrax::TextDocument> pairs;
    std::istringstream input(astrax::embedded::kDialoguePairs);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::size_t separator = line.find('\t');
        if (separator == std::string::npos) {
            continue;
        }
        pairs.push_back({{}, line.substr(0, separator),
                         line.substr(separator + 1)});
    }
    return pairs;
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

std::wstring wide_from_utf8(const std::string& value) {
#ifdef _WIN32
    if (value.empty()) {
        return {};
    }
    int size = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), nullptr, 0);
    UINT flags = MB_ERR_INVALID_CHARS;
    if (size <= 0) {
        // A model checkpoint can contain legacy malformed bytes. The
        // dialogue decoder sanitizes current output, but the console boundary
        // must also tolerate old checkpoints and never terminate the session.
        flags = 0;
        size = MultiByteToWideChar(
            CP_UTF8, flags, value.data(), static_cast<int>(value.size()),
            nullptr, 0);
    }
    if (size <= 0) {
        return std::wstring(value.size(), L'\uFFFD');
    }
    std::wstring result(static_cast<std::size_t>(size), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8, flags, value.data(),
            static_cast<int>(value.size()), result.data(), size) <= 0) {
        return std::wstring(value.size(), L'\uFFFD');
    }
    return result;
#else
    return std::wstring(value.begin(), value.end());
#endif
}

void write_utf8(const std::string& value) {
#ifdef _WIN32
    DWORD mode = 0;
    const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    if (output != INVALID_HANDLE_VALUE && GetConsoleMode(output, &mode)) {
        // wide_from_utf8 validates the complete response and falls back to
        // Windows' replacement behavior for malformed legacy bytes.  Never
        // concatenate a code-page string here: that was the source of the
        // previous "cannot convert UTF-8 text" failure path.
        const std::wstring wide = wide_from_utf8(value);
        DWORD written = 0;
        if (!WriteConsoleW(output, wide.data(),
                           static_cast<DWORD>(wide.size()), &written, nullptr)) {
            throw std::runtime_error("cannot write to Windows console");
        }
        return;
    }
    // stdout may be a pipe or a redirected file. The wide entry point can
    // leave the CRT stream in Unicode mode, so write UTF-8 bytes directly
    // instead of routing them through std::cout.
    const HANDLE redirected = GetStdHandle(STD_OUTPUT_HANDLE);
    if (redirected != INVALID_HANDLE_VALUE) {
        const char* cursor = value.data();
        std::size_t remaining = value.size();
        while (remaining > 0U) {
            const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(
                remaining, static_cast<std::size_t>(0x7ffff000U)));
            DWORD written = 0;
            if (!WriteFile(redirected, cursor, chunk, &written, nullptr)) {
                throw std::runtime_error("cannot write UTF-8 output");
            }
            cursor += written;
            remaining -= written;
            if (written == 0U) {
                throw std::runtime_error("cannot make progress writing UTF-8 output");
            }
        }
        return;
    }
#else
    std::cout << value;
#endif
}

bool read_console_line(std::wstring& line) {
#ifdef _WIN32
    DWORD mode = 0;
    const HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    if (input != INVALID_HANDLE_VALUE && GetConsoleMode(input, &mode)) {
        wchar_t buffer[8192]{};
        DWORD read = 0;
        if (!ReadConsoleW(input, buffer, 8191, &read, nullptr)) {
            return false;
        }
        line.assign(buffer, buffer + read);
        while (!line.empty() &&
               (line.back() == L'\r' || line.back() == L'\n')) {
            line.pop_back();
        }
        return true;
    }
#endif
    return static_cast<bool>(std::getline(std::wcin, line));
}

} // namespace

int wmain(int argc, wchar_t** argv) {
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        astrax::AstraxModel model;
        model.load_checkpoint_bytes(astrax::embedded::kCheckpoint);
        model.set_goal(
            {"conversation", "Understand the complete message and answer it", 0.9F, true});

        if (argc > 1) {
            std::string message = utf8_from_wide(argv[1]);
            for (int index = 2; index < argc; ++index) {
                message += " ";
                message += utf8_from_wide(argv[index]);
            }
            write_utf8(model.chat(message) + "\n");
            return 0;
        }

        write_utf8(u8"Astrax dialogue ready. \u8F93\u5165 /exit \u7ED3\u675F\u3002\n");
        std::wstring wide_message;
        while (true) {
            write_utf8(u8"\u4F60> ");
            if (!read_console_line(wide_message) ||
                wide_message == L"/exit") {
                break;
            }
            const std::string message = utf8_from_wide(wide_message);
            if (message.empty()) {
                continue;
            }
            write_utf8("Astrax> " + model.chat(message) + "\n");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "chat failure: " << error.what() << '\n';
        return 1;
    }
}
