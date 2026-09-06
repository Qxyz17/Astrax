#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "astrax/math.hpp"

namespace astrax {

struct DialogueExample {
    std::string input;
    std::string response;
};

struct DialogueTrainingReport {
    std::size_t examples = 0;
    std::size_t epochs = 0;
    float cross_entropy = 0.0F;
    float byte_accuracy = 0.0F;
    float exact_match = 0.0F;
};

class HolisticDialogueModel {
public:
    HolisticDialogueModel(std::size_t input_dim = 96,
                          std::size_t max_response_bytes = 256,
                          float learning_rate = 0.08F);

    DialogueTrainingReport train(const std::vector<DialogueExample>& dataset,
                                 std::size_t epochs);
    std::string respond(const std::string& input,
                        float* confidence = nullptr) const;

    void save(std::ostream& output) const;
    void load(std::istream& input);

    static void validate_dataset(const std::vector<DialogueExample>& dataset,
                                 std::size_t max_response_bytes);
    static void save_tsv(const std::string& path,
                         const std::vector<DialogueExample>& dataset);
    static std::vector<DialogueExample> load_tsv(const std::string& path,
                                                 std::size_t max_response_bytes);

    bool trained() const noexcept { return trained_; }
    std::size_t input_dim() const noexcept { return input_dim_; }
    std::size_t max_response_bytes() const noexcept {
        return max_response_bytes_;
    }

private:
    math::Vector encode_input(const std::string& input) const;
    std::size_t weight_index(std::size_t slot,
                             std::size_t byte,
                             std::size_t feature) const noexcept;

    std::size_t input_dim_;
    std::size_t max_response_bytes_;
    float learning_rate_;
    math::Vector weights_;
    math::Vector bias_;
    bool trained_ = false;
};

} // namespace astrax
