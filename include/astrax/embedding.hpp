#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "astrax/math.hpp"

namespace astrax {

// Input-side subword encoder. The output side of Astrax keeps the complete
// Unicode codepoint space; subwords only give the input a learned, shared
// representation so synonyms, morphology, and mixed Chinese/English text are
// not scattered across unrelated hash buckets.
class SubwordEncoder {
public:
    SubwordEncoder() = default;

    // Loads the binary merge table written by tools/bpe/learn_bpe.py.
    static SubwordEncoder load(const std::string& path);
    static SubwordEncoder from_bytes(const std::vector<std::uint8_t>& bytes);

    // Splits text into subword ids. Unknown codepoints become their own id.
    std::vector<std::uint32_t> encode_ids(const std::string& text) const;

    void save(std::ostream& output) const;
    void load(std::istream& input);

    std::size_t vocabulary_size() const noexcept { return vocabulary_.size(); }
    bool empty() const noexcept { return merges_.empty(); }

private:
    // Pairs of token ids merged into a new token id.
    struct Merge {
        std::uint32_t left = 0;
        std::uint32_t right = 0;
        std::uint32_t result = 0;
    };

    std::uint32_t token_for(const std::string& token);

    std::vector<Merge> merges_;
    // token id -> readable text (first-entry wins for reporting/debug).
    std::vector<std::string> vocabulary_;
    std::unordered_map<std::uint32_t, std::uint32_t> base_ids_;
    std::uint32_t next_id_ = 1;
};

// Trainable embedding table over subword ids. The table participates in
// training; pooling turns a sequence of ids into one fixed-size vector.
class EmbeddingTable {
public:
    EmbeddingTable() = default;
    EmbeddingTable(std::size_t dimension, std::size_t capacity);

    // Mean-pools the embeddings of the given ids. Unknown ids are skipped.
    math::Vector encode(const std::vector<std::uint32_t>& ids) const;

    std::size_t dimension() const noexcept { return dimension_; }
    std::size_t capacity() const noexcept { return capacity_; }
    const math::Vector& weights() const noexcept { return weights_; }
    math::Vector& weights() noexcept { return weights_; }

    void save(std::ostream& output) const;
    void load(std::istream& input);

private:
    std::size_t dimension_ = 0;
    std::size_t capacity_ = 0;
    math::Vector weights_; // capacity_ * dimension_
};

} // namespace astrax
