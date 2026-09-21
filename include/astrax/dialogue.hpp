#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "astrax/math.hpp"

namespace astrax {

// The text-training unit. A document is trained with a masked-document
// reconstruction objective (whole-sequence denoising). An input/target pair is
// used only for conditional fine-tuning after document pretraining.
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

// Holistic iterative-denoising text model.
//
// This model is deliberately not a Transformer and not a next-token
// autoregressive decoder. A fixed number of parallel response slots are
// initialized and then refined over several passes. On every pass each slot
// reads the complete encoded input, the condition vector, and an aggregate of
// the previous pass's slot distributions. No slot reads another slot's output
// within the same pass, and no pass produces tokens left to right. The final
// pass yields the complete sequence at once.
class HolisticDialogueModel {
public:
    HolisticDialogueModel(std::size_t input_dim = 512,
                          std::size_t max_response_codepoints = 128,
                          float learning_rate = 0.05F,
                          std::size_t condition_dim = 128);

    // Stage A: masked-document reconstruction over complete documents.
    //
    // When conditions is non-empty it must have one entry per document. Each
    // entry is the Osten-derived condition for that document, so training
    // learns how the model's own thinking state shapes what it says. When
    // conditions is empty the zero condition is used.
    DialogueTrainingReport train(const std::vector<TextDocument>& dataset,
                                 std::size_t epochs,
                                 const std::vector<math::Vector>& conditions = {});

    // Stage B: conditional fine-tuning on input/target pairs. Continues from
    // the parameters learned by train(); no retrieval or nearest-neighbor
    // lookup is performed at inference.
    DialogueTrainingReport train_pairs(const std::vector<TextDocument>& dataset,
                                       std::size_t epochs,
                                       const std::vector<math::Vector>& conditions = {});
    DialogueTrainingReport evaluate_pairs(const std::vector<TextDocument>& dataset) const;

    // Builds a condition vector from an Osten state and goal. Exposed so the
    // owner (AstraxModel) can compute per-document training conditions from
    // real Osten forward passes.
    math::Vector make_condition(const math::Vector& osten_state,
                                std::size_t action_id,
                                float action_value,
                                const math::Vector& goal,
                                const std::string& memory_context) const {
        return encode_condition(osten_state, action_id, action_value, goal,
                                memory_context);
    }

    // Every output slot reads the complete encoded input and condition vector.
    // The response is produced by iterative whole-sequence refinement.
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

    // Factorized per-slot logits: a high-part distribution and a low-part
    // distribution. Their combination yields a full Unicode codepoint.
    struct SlotLogits {
        math::Vector high; // high_count_
        math::Vector low;  // low_count_
    };

    // One refinement pass. Reads the aggregate of the previous pass's slot
    // distributions and returns the new per-slot factorized logits. Slot and
    // round embeddings select per-position and per-pass parameters.
    void forward_pass(const math::Vector& features,
                      const math::Vector& condition,
                      const math::Vector& aggregate,
                      std::size_t round,
                      std::vector<SlotLogits>& logits) const;

    // Aggregates all slot distributions into a fixed-size vector. This is the
    // only channel through which slots exchange information between passes.
    math::Vector aggregate_slots(const std::vector<SlotLogits>& logits) const;

    // Computes softmax probabilities for a vector of logits.
    static math::Vector softmax(const math::Vector& logits);

    // Trains a single refinement pass against per-slot target codepoints using
    // manual backpropagation. Returns the mean loss over active slots.
    float train_pass(const math::Vector& features,
                     const math::Vector& condition,
                     const math::Vector& aggregate,
                     std::size_t round,
                     const std::vector<std::uint32_t>& targets,
                     const std::vector<char>& active);

    // A complete copy of every learnable parameter. Used for data-parallel
    // training: each worker trains a private replica from the same snapshot,
    // and the replicas are averaged back after each epoch.
    struct Replica {
        math::Vector in_weights;
        math::Vector cond_weights;
        math::Vector slot_weights;
        math::Vector round_weights;
        math::Vector agg_weights;
        math::Vector hidden_bias;
        math::Vector out_high_weights;
        math::Vector out_high_bias;
        math::Vector out_low_weights;
        math::Vector out_low_bias;
        math::Vector slot_embedding_table;
        math::Vector round_embedding_table;
    };

    Replica capture() const;
    void install(const Replica& replica);
    void accumulate(const Replica& replica);
    // Trains the documents in [begin, end) for one epoch against the current
    // parameters and returns the loss and accuracy accumulated for that range.
    void train_range(const std::vector<TextDocument>& dataset,
                     const std::vector<math::Vector>& conditions,
                     std::size_t begin, std::size_t end,
                     float& loss, std::size_t& total_slots,
                     std::size_t& correct_slots);

    std::size_t input_dim_;
    std::size_t max_response_codepoints_;
    float learning_rate_;
    std::size_t condition_dim_;
    std::size_t hidden_dim_ = 192;
    std::size_t slot_dim_ = 64;
    std::size_t round_dim_ = 32;
    std::size_t agg_dim_ = 128;
    // Iterative whole-sequence refinement passes. More than one pass is
    // required for the coarse-to-fine denoising contract, but each extra pass
    // multiplies training cost, so keep it small.
    std::size_t rounds_ = 3;

    // The complete Unicode codepoint space is factorized into a high part and
    // a low part. There is no data-derived vocabulary and no truncation: any
    // UTF-8 codepoint can be selected. See astrax/charset.hpp.
    std::size_t high_count_ = 0;
    std::size_t low_count_ = 0;

    // input -> hidden
    math::Vector in_weights_;   // hidden_dim_ * input_dim_
    // condition -> hidden
    math::Vector cond_weights_; // hidden_dim_ * condition_dim_
    // slot embedding -> hidden
    math::Vector slot_weights_; // hidden_dim_ * slot_dim_
    // round embedding -> hidden
    math::Vector round_weights_;// hidden_dim_ * round_dim_
    // aggregate -> hidden
    math::Vector agg_weights_;  // hidden_dim_ * agg_dim_
    math::Vector hidden_bias_;  // hidden_dim_
    // hidden -> factorized codepoint classes
    math::Vector out_high_weights_;  // high_count_ * hidden_dim_
    math::Vector out_high_bias_;     // high_count_
    math::Vector out_low_weights_;   // low_count_ * hidden_dim_
    math::Vector out_low_bias_;      // low_count_

    // Learned embeddings (not projected directly; used as one-hot sources).
    math::Vector slot_embedding_table_;  // max_response_codepoints_ * slot_dim_
    math::Vector round_embedding_table_; // rounds_ * round_dim_

    bool trained_ = false;
};

} // namespace astrax
