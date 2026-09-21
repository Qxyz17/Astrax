#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include "astrax/dialogue.hpp"

int main() {
    astrax::HolisticDialogueModel model(256, 64, 0.08F, 128);
    std::vector<astrax::TextDocument> docs;
    std::string line;
    FILE* f = std::fopen("data/wiki_en.txt", "rb");
    char buf[8192];
    int count = 0;
    while (count < 50 && std::fgets(buf, sizeof(buf), f)) {
        line = buf;
        if (line.empty() || line[0] == '#') continue;
        docs.push_back({{}, line, line});
        ++count;
    }
    std::fclose(f);
    auto t0 = std::chrono::steady_clock::now();
    auto report = model.train_pairs(docs, 1);
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::printf("docs=%zu seconds=%.2f sec_per_doc=%.3f accuracy=%.4f\n",
                docs.size(), sec, sec / (double)docs.size(), report.codepoint_accuracy);
    return 0;
}
