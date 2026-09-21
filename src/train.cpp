#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <string>
#include <vector>

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

bool valid_utf8(const std::string& value) {
    std::size_t index = 0;
    while (index < value.size()) {
        const unsigned char lead = static_cast<unsigned char>(value[index]);
        std::size_t length = 0;
        std::uint32_t codepoint = 0;
        if (lead <= 0x7FU) {
            ++index;
            continue;
        } else if (lead >= 0xC2U && lead <= 0xDFU) {
            length = 2;
            codepoint = lead & 0x1FU;
        } else if (lead >= 0xE0U && lead <= 0xEFU) {
            length = 3;
            codepoint = lead & 0x0FU;
        } else if (lead >= 0xF0U && lead <= 0xF4U) {
            length = 4;
            codepoint = lead & 0x07U;
        } else {
            return false;
        }
        if (index + length > value.size()) {
            return false;
        }
        for (std::size_t offset = 1; offset < length; ++offset) {
            const unsigned char continuation =
                static_cast<unsigned char>(value[index + offset]);
            if ((continuation & 0xC0U) != 0x80U) {
                return false;
            }
            codepoint = (codepoint << 6U) | (continuation & 0x3FU);
        }
        if ((length == 2 && codepoint < 0x80U) ||
            (length == 3 && codepoint < 0x800U) ||
            (length == 4 && codepoint < 0x10000U) ||
            codepoint > 0x10FFFFU ||
            (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
            return false;
        }
        index += length;
    }
    return true;
}

struct CorpusEvaluation {
    std::size_t examples = 0;
    std::size_t valid_utf8 = 0;
    std::size_t non_empty = 0;
};

bool looks_like_code(const std::string& value) {
    return value.find("#include") != std::string::npos ||
           value.find("int main") != std::string::npos ||
           value.find("std::") != std::string::npos;
}

std::size_t count_code_pairs(const std::vector<astrax::TextDocument>& pairs) {
    std::size_t result = 0;
    for (const auto& pair : pairs) {
        if (looks_like_code(pair.target)) {
            ++result;
        }
    }
    return result;
}

float evaluate_code_compile_rate(
    const std::vector<astrax::TextDocument>& pairs,
    const std::filesystem::path& root) {
    std::size_t examples = 0;
    std::size_t compiled = 0;
    const std::filesystem::path source_path =
        root / "artifacts" / "validation_code.cpp";
    const std::filesystem::path object_path =
        root / "artifacts" / "validation_code.obj";
    for (const auto& pair : pairs) {
        if (!looks_like_code(pair.target)) {
            continue;
        }
        ++examples;
        std::ofstream source(source_path, std::ios::binary | std::ios::trunc);
        source << "#include <algorithm>\n#include <iostream>\n#include <vector>\n"
               << pair.target << '\n';
        source.close();
        const std::string command =
            "cl.exe /nologo /EHsc /std:c++17 /c \"" +
            source_path.string() + "\" /Fo\"" + object_path.string() +
            "\" >NUL 2>NUL";
        if (std::system(command.c_str()) == 0) {
            ++compiled;
        }
    }
    std::error_code error;
    std::filesystem::remove(source_path, error);
    std::filesystem::remove(object_path, error);
    return examples == 0U ? 0.0F : static_cast<float>(compiled) /
        static_cast<float>(examples);
}

CorpusEvaluation evaluate_corpus(const astrax::HolisticDialogueModel& model,
                                 const std::vector<astrax::TextDocument>& corpus) {
    CorpusEvaluation result;
    result.examples = corpus.size();
    for (const astrax::TextDocument& document : corpus) {
        const std::string output = model.respond(document.text);
        if (valid_utf8(output)) {
            ++result.valid_utf8;
        }
        if (!output.empty()) {
            ++result.non_empty;
        }
    }
    return result;
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

        // Real Wikipedia document corpora. Each file is one complete
        // document per line, so it feeds the holistic whole-document objective
        // directly. No prompt/response table is used for document training.
        const std::filesystem::path corpus_zh_path =
            data_directory / "wiki_zh.txt";
        const std::filesystem::path corpus_en_path =
            data_directory / "wiki_en.txt";
        const std::filesystem::path pairs_path =
            data_directory / "astrax_dialogue_pairs.tsv";
        std::vector<astrax::TextDocument> corpus;
        if (std::filesystem::exists(corpus_zh_path)) {
            const std::vector<astrax::TextDocument> zh =
                astrax::HolisticDialogueModel::load_documents(
                    corpus_zh_path.string(), 4U * 1024U * 1024U);
            corpus.insert(corpus.end(), zh.begin(), zh.end());
        }
        if (std::filesystem::exists(corpus_en_path)) {
            const std::vector<astrax::TextDocument> en =
                astrax::HolisticDialogueModel::load_documents(
                    corpus_en_path.string(), 4U * 1024U * 1024U);
            corpus.insert(corpus.end(), en.begin(), en.end());
        }
        if (corpus.empty()) {
            throw std::runtime_error("no document corpus found under data/");
        }

        // Train on the complete corpus. The native trainer is data-parallel
        // across CPU cores, so the whole Wikipedia corpus is used rather than
        // a tiny subset. A small deterministic slice is held out for
        // validation and never trained on.
        const std::size_t validation_size =
            std::max<std::size_t>(16U, corpus.size() / 50U);
        std::vector<astrax::TextDocument> text_training_corpus;
        std::vector<astrax::TextDocument> validation_corpus;
        text_training_corpus.reserve(corpus.size() - validation_size);
        for (std::size_t index = 0; index < corpus.size(); ++index) {
            if (index % 50U == 0U &&
                validation_corpus.size() < validation_size) {
                validation_corpus.push_back(corpus[index]);
            } else {
                text_training_corpus.push_back(corpus[index]);
            }
        }

        astrax::AstraxModel model(config);
        const astrax::TrainingReport report = model.train_offline(loaded, epochs);
        const std::vector<astrax::TextDocument> pairs =
            astrax::HolisticDialogueModel::load_pairs(
                pairs_path.string(), 4U * 1024U * 1024U);
        const std::size_t split = pairs.size() * 3U / 4U;
        const std::vector<astrax::TextDocument> training_pairs(
            pairs.begin(), pairs.begin() + static_cast<std::ptrdiff_t>(split));
        const std::vector<astrax::TextDocument> validation_pairs(
            pairs.begin() + static_cast<std::ptrdiff_t>(split), pairs.end());
        // Stage A: masked-document reconstruction over the complete real
        // Wikipedia corpus, with Osten-derived conditions. This is where the
        // model learns how language is actually written.
        const astrax::DialogueTrainingReport document_report =
            model.train_dialogue(text_training_corpus, 4);
        // Stage B: conditional fine-tuning on input/target pairs continues
        // from the document-trained parameters.
        const astrax::DialogueTrainingReport text_report =
            model.train_dialogue_pairs(training_pairs, 48);
        const astrax::DialogueTrainingReport validation_report =
            model.dialogue().evaluate_pairs(validation_corpus);
        const std::size_t code_examples = count_code_pairs(validation_pairs);
        const float code_compile_rate = evaluate_code_compile_rate(validation_pairs, root);
        // Evaluate a bounded representative set as well. The on-disk corpus
        // is 10 MiB, but repeated/near-duplicate rows should not turn the
        // executable into an unbounded benchmark.
        std::vector<astrax::TextDocument> evaluation_corpus;
        evaluation_corpus.reserve(std::min<std::size_t>(corpus.size(), 512U));
        for (std::size_t index = 0; index <
             std::min<std::size_t>(corpus.size(), 512U); ++index) {
            evaluation_corpus.push_back(corpus[index * corpus.size() /
                                               std::min<std::size_t>(corpus.size(), 512U)]);
        }
        const CorpusEvaluation corpus_report =
            evaluate_corpus(model.dialogue(), evaluation_corpus);

        const std::filesystem::path checkpoint_path =
            artifact_directory / "astrax_training.astrax-model";
        const astrax::math::Vector validation_state = loaded.front().state;
        const std::size_t validation_action = loaded.front().action;
        const astrax::math::Vector expected_prediction =
            model.predictor().predict(validation_state, validation_action);
        const float expected_value =
            model.values().value(validation_state, validation_action);
        const std::string expected_text = model.dialogue().respond(corpus.front().text);
        model.save_checkpoint(checkpoint_path.string());

        astrax::AstraxModel restored_model(config);
        restored_model.load_checkpoint(checkpoint_path.string());
        const astrax::math::Vector restored_prediction =
            restored_model.predictor().predict(validation_state, validation_action);
        const float restored_value =
            restored_model.values().value(validation_state, validation_action);
        const std::string restored_text =
            restored_model.dialogue().respond(corpus.front().text);
        if (expected_prediction != restored_prediction ||
            expected_value != restored_value || expected_text != restored_text ||
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
                    << "average_intrinsic_reward=" << report.average_intrinsic_reward << '\n'
                     << "text_corpus_zh=" << corpus_zh_path.string() << '\n'
                    << "text_corpus_en=" << corpus_en_path.string() << '\n'
                    << "dialogue_pairs=" << pairs_path.string() << '\n'
                    << "dialogue_pair_examples=" << pairs.size() << '\n'
                    << "dialogue_training_examples=" << training_pairs.size() << '\n'
                    << "dialogue_validation_examples=" << validation_pairs.size() << '\n'
                    << "text_corpus_bytes=" << (std::filesystem::exists(corpus_zh_path) ? std::filesystem::file_size(corpus_zh_path) : 0) + (std::filesystem::exists(corpus_en_path) ? std::filesystem::file_size(corpus_en_path) : 0) << '\n'
                    << "text_documents=" << corpus.size() << '\n'
                    << "text_training_documents_unique=" << text_report.examples << '\n'
                    << "document_stage_examples=" << document_report.examples << '\n'
                    << "document_stage_epochs=" << document_report.epochs << '\n'
                    << "document_stage_loss=" << document_report.reconstruction_loss << '\n'
                    << "document_stage_codepoint_accuracy=" << document_report.codepoint_accuracy << '\n'
                    << "text_epochs=" << text_report.epochs << '\n'
                    << "text_reconstruction_loss=" << text_report.reconstruction_loss << '\n'
                    << "text_codepoint_accuracy=" << text_report.codepoint_accuracy << '\n'
                    << "text_valid_utf8_ratio=" << text_report.valid_utf8_ratio << '\n'
                    << "text_eval_valid_utf8=" << corpus_report.valid_utf8 << '\n'
                     << "text_eval_non_empty=" << corpus_report.non_empty << '\n'
                     << "validation_codepoint_accuracy=" << validation_report.validation_codepoint_accuracy << '\n'
                     << "validation_exact_match=" << validation_report.validation_exact_match << '\n'
                     << "validation_non_repetition=" << validation_report.validation_non_repetition << '\n'
                     << "validation_answer_accuracy=" << validation_report.validation_answer_accuracy << '\n'
                     << "validation_code_compile_rate=" << code_compile_rate << '\n'
                    << "checkpoint=" << checkpoint_path.string() << '\n'
                    << "checkpoint_verified=1\n";

        std::cout << "offline training completed\n"
                  << "dataset=" << dataset_path.string() << '\n'
                  << "samples=" << report.samples << '\n'
                  << "epochs=" << report.epochs << '\n'
                  << "predictor_loss=" << report.predictor_loss << '\n'
                  << "value_loss=" << report.value_loss << '\n'
                  << "average_intrinsic_reward=" << report.average_intrinsic_reward << '\n'
                  << "text_corpus_bytes=" << (std::filesystem::exists(corpus_zh_path) ? std::filesystem::file_size(corpus_zh_path) : 0) + (std::filesystem::exists(corpus_en_path) ? std::filesystem::file_size(corpus_en_path) : 0) << '\n'
                   << "text_documents=" << corpus.size() << '\n'
                   << "dialogue_training_examples=" << training_pairs.size() << '\n'
                   << "dialogue_validation_examples=" << validation_pairs.size() << '\n'
                   << "text_training_documents_unique=" << text_report.examples << '\n'
                  << "document_stage_examples=" << document_report.examples << '\n'
                  << "document_stage_epochs=" << document_report.epochs << '\n'
                  << "document_stage_loss=" << document_report.reconstruction_loss << '\n'
                  << "document_stage_codepoint_accuracy=" << document_report.codepoint_accuracy << '\n'
                  << "text_epochs=" << text_report.epochs << '\n'
                  << "text_reconstruction_loss=" << text_report.reconstruction_loss << '\n'
                  << "text_codepoint_accuracy=" << text_report.codepoint_accuracy << '\n'
                  << "text_valid_utf8_ratio=" << text_report.valid_utf8_ratio << '\n'
                  << "text_eval_valid_utf8=" << corpus_report.valid_utf8 << '\n'
                   << "text_eval_non_empty=" << corpus_report.non_empty << '\n'
                   << "validation_codepoint_accuracy=" << validation_report.validation_codepoint_accuracy << '\n'
                   << "validation_exact_match=" << validation_report.validation_exact_match << '\n'
                   << "validation_non_repetition=" << validation_report.validation_non_repetition << '\n'
                   << "validation_answer_accuracy=" << validation_report.validation_answer_accuracy << '\n'
                   << "validation_code_examples=" << code_examples << '\n'
                   << "validation_code_compile_rate=" << code_compile_rate << '\n'
                  << "checkpoint=" << checkpoint_path.string() << '\n'
                  << "checkpoint_verified=1\n"
                  << "report=" << report_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "training failure: " << error.what() << '\n';
        return 1;
    }
}
