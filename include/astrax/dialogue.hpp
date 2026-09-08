#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "astrax/math.hpp"

namespace astrax {

// A document is the only text-training unit. There is deliberately no
// input/response pair in this API: the model learns a denoising objective over
// complete text blocks.
struct TextDocument {
    std::string text;
    std::string input;
    std::string target;
};

struct DialogueTrainingReport {
    std::size_t examples = 0;
    std::size_t epochs = 0;
    float reconstruction_loss = 0.0F;
    float response_loss = 0.0F;
    float cross_entropy = 0.0F; // source compatibility with 0.1.x
    float codepoint_accuracy = 0.0F;
    float byte_accuracy = 0.0F; // source compatibility with 0.1.x
    float exact_match = 0.0F;
    float valid_utf8_ratio = 0.0F;
    float validation_codepoint_accuracy = 0.0F;
    float validation_exact_match = 0.0F;
    float validation_non_repetition = 0.0F;
    float validation_code_compile_rate = 0.0F;
    float validation_answer_accuracy = 0.0F;
};

class HolisticDialogueModel {
public:
    HolisticDialogueModel(std::size_t input_dim = 192,
                          std::size_t max_response_codepoints = 64,
                          float learning_rate = 0.02F,
                          std::size_t condition_dim = 64);

    DialogueTrainingReport train(const std::vector<TextDocument>& dataset,
                                 std::size_t epochs);
    DialogueTrainingReport train_pairs(const std::vector<TextDocument>& dataset,
                                       std::size_t epochs);
    DialogueTrainingReport evaluate_pairs(const std::vector<TextDocument>& dataset) const;

    // Every output slot reads the complete encoded input and condition vector.
    // No slot reads another emitted slot.
    std::string respond(const std::string& input,
                        const math::Vector& osten_state = {},
                        std::size_t action_id = 0,
                        float action_value = 0.0F,
                        const math::Vector& goal = {},
                        const std::string& memory_context = {},
                        float* confidence = nullptr) const;

    void save(std::ostream& output) const;
    void load(std::istream& input);

    static void validate_dataset(const std::vector<TextDocument>& dataset,
                                 std::size_t max_document_bytes);
    static std::vector<TextDocument> load_documents(
        const std::string& path, std::size_t max_document_bytes);
    static std::vector<TextDocument> load_pairs(
        const std::string& path, std::size_t max_document_bytes);

    bool trained() const noexcept { return trained_; }
    std::size_t input_dim() const noexcept { return input_dim_; }
    std::size_t max_response_codepoints() const noexcept {
        return max_response_codepoints_;
    }
    std::size_t max_response_bytes() const noexcept {
        return max_response_codepoints_ * 4;
    }

private:
    math::Vector encode_text(const std::string& input) const;
    math::Vector encode_condition(const math::Vector& osten_state,
                                  std::size_t action_id,
                                  float action_value,
                                  const math::Vector& goal,
                                  const std::string& memory_context) const;
    math::Vector hidden_for(const math::Vector& features,
                            const math::Vector& condition,
                            std::size_t slot) const;

    std::size_t input_dim_;
    std::size_t max_response_codepoints_;
    float learning_rate_;
    std::size_t condition_dim_;
    std::size_t hidden_dim_;
    std::vector<std::uint32_t> vocabulary_;
    math::Vector context_weights_;
    math::Vector context_bias_;
    math::Vector condition_weights_;
    math::Vector slot_embeddings_;
    math::Vector class_embeddings_;
    math::Vector class_bias_;
    bool trained_ = false;
};

} // namespace astrax
