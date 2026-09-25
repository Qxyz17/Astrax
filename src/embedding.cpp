#include "astrax/embedding.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace astrax {
namespace {

std::vector<std::uint32_t> decode_codepoints(const std::string& value) {
    std::vector<std::uint32_t> result;
    std::size_t index = 0;
    while (index < value.size()) {
        const unsigned char lead = static_cast<unsigned char>(value[index]);
        std::size_t length = 0;
        std::uint32_t codepoint = 0;
        if (lead <= 0x7FU) {
            length = 1;
            codepoint = lead;
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
            ++index;
            continue;
        }
        if (index + length > value.size()) {
            break;
        }
        bool valid = true;
        for (std::size_t offset = 1; offset < length; ++offset) {
            const unsigned char continuation =
                static_cast<unsigned char>(value[index + offset]);
            if ((continuation & 0xC0U) != 0x80U) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6U) | (continuation & 0x3FU);
        }
        if (valid) {
            result.push_back(codepoint);
        }
        index += length;
    }
    return result;
}

void write_size(std::ostream& output, std::size_t value) {
    const std::uint64_t stored = static_cast<std::uint64_t>(value);
    output.write(reinterpret_cast<const char*>(&stored), sizeof(stored));
}

std::size_t read_size(std::istream& input, std::size_t limit = 100000000U) {
    std::uint64_t stored = 0;
    input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
    if (!input || stored > limit) {
        throw std::runtime_error("invalid embedding checkpoint size");
    }
    return static_cast<std::size_t>(stored);
}

} // namespace

SubwordEncoder SubwordEncoder::from_bytes(const std::vector<std::uint8_t>& bytes) {
    SubwordEncoder encoder;
    std::size_t cursor = 0;
    constexpr char magic[] = "ASTRAX-BPE-1";
    const std::size_t magic_size = sizeof(magic) - 1U;
    if (bytes.size() < magic_size ||
        std::string(reinterpret_cast<const char*>(bytes.data()), magic_size) !=
            std::string(magic, magic_size)) {
        throw std::runtime_error("subword table has an unexpected header");
    }
    cursor += magic_size;
    if (cursor + 4U > bytes.size()) {
        throw std::runtime_error("subword table is truncated");
    }
    std::uint32_t merge_count = 0;
    std::memcpy(&merge_count, bytes.data() + cursor, sizeof(merge_count));
    cursor += sizeof(merge_count);

    encoder.merges_.reserve(merge_count);
    for (std::uint32_t index = 0; index < merge_count; ++index) {
        auto read_token = [&](std::string& out) {
            if (cursor + 2U > bytes.size()) {
                throw std::runtime_error("subword table ended in a token length");
            }
            std::uint16_t length = 0;
            std::memcpy(&length, bytes.data() + cursor, sizeof(length));
            cursor += sizeof(length);
            if (cursor + length > bytes.size()) {
                throw std::runtime_error("subword table ended in a token body");
            }
            out.assign(reinterpret_cast<const char*>(bytes.data() + cursor), length);
            cursor += length;
        };
        std::string left;
        std::string right;
        read_token(left);
        read_token(right);
        Merge merge;
        merge.left = encoder.token_for(left);
        merge.right = encoder.token_for(right);
        const std::string combined = left + right;
        merge.result = encoder.token_for(combined);
        encoder.merges_.push_back(merge);
    }
    return encoder;
}

SubwordEncoder SubwordEncoder::load(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open subword table: " + path);
    }
    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    return from_bytes(bytes);
}

std::uint32_t SubwordEncoder::token_for(const std::string& token) {
    const std::vector<std::uint32_t> codepoints = decode_codepoints(token);
    if (codepoints.size() == 1U) {
        const std::uint32_t codepoint = codepoints.front();
        auto found = base_ids_.find(codepoint);
        if (found != base_ids_.end()) {
            return found->second;
        }
        const std::uint32_t id = next_id_++;
        base_ids_[codepoint] = id;
        if (vocabulary_.size() <= id) {
            vocabulary_.resize(id + 1U);
        }
        vocabulary_[id] = token;
        return id;
    }
    // A multi-codepoint token that is not yet known gets a fresh id. Its
    // composition is recorded by the merge that produced it.
    const std::uint32_t id = next_id_++;
    if (vocabulary_.size() <= id) {
        vocabulary_.resize(id + 1U);
    }
    vocabulary_[id] = token;
    return id;
}

std::vector<std::uint32_t> SubwordEncoder::encode_ids(
    const std::string& text) const {
    std::vector<std::uint32_t> ids;
    for (const std::uint32_t codepoint : decode_codepoints(text)) {
        auto found = base_ids_.find(codepoint);
        if (found != base_ids_.end()) {
            ids.push_back(found->second);
        }
    }
    // Apply merges greedily by merge order (most frequent first).
    for (const Merge& merge : merges_) {
        for (std::size_t index = 0; index + 1U < ids.size();) {
            if (ids[index] == merge.left && ids[index + 1U] == merge.right) {
                ids[index] = merge.result;
                ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(index) + 1);
            } else {
                ++index;
            }
        }
    }
    return ids;
}

void SubwordEncoder::save(std::ostream& output) const {
    write_size(output, merges_.size());
    for (const Merge& merge : merges_) {
        write_size(output, merge.left);
        write_size(output, merge.right);
        write_size(output, merge.result);
    }
    write_size(output, vocabulary_.size());
    for (const std::string& token : vocabulary_) {
        write_size(output, token.size());
        output.write(token.data(), static_cast<std::streamsize>(token.size()));
    }
    if (!output) {
        throw std::runtime_error("failed to write subword encoder");
    }
}

void SubwordEncoder::load(std::istream& input) {
    merges_.clear();
    vocabulary_.clear();
    base_ids_.clear();
    next_id_ = 1;
    const std::size_t merge_count = read_size(input);
    merges_.resize(merge_count);
    for (Merge& merge : merges_) {
        merge.left = static_cast<std::uint32_t>(read_size(input));
        merge.right = static_cast<std::uint32_t>(read_size(input));
        merge.result = static_cast<std::uint32_t>(read_size(input));
    }
    const std::size_t vocabulary_size = read_size(input);
    vocabulary_.resize(vocabulary_size);
    for (std::string& token : vocabulary_) {
        const std::size_t size = read_size(input);
        token.assign(size, '\0');
        input.read(token.data(), static_cast<std::streamsize>(size));
    }
    if (!input) {
        throw std::runtime_error("subword encoder checkpoint is incomplete");
    }
    next_id_ = static_cast<std::uint32_t>(vocabulary_.size());
    for (std::size_t id = 1; id < vocabulary_.size(); ++id) {
        const std::vector<std::uint32_t> codepoints =
            decode_codepoints(vocabulary_[id]);
        if (codepoints.size() == 1U) {
            base_ids_[codepoints.front()] = static_cast<std::uint32_t>(id);
        }
    }
}

EmbeddingTable::EmbeddingTable(std::size_t dimension, std::size_t capacity)
    : dimension_(dimension), capacity_(capacity) {
    if (dimension_ == 0 || capacity_ == 0) {
        throw std::invalid_argument("embedding dimensions must be positive");
    }
    weights_.assign(capacity_ * dimension_, 0.0F);
    // Small deterministic initialization so training can move every entry.
    for (std::size_t index = 0; index < weights_.size(); ++index) {
        const float phase = static_cast<float>((index * 37U + 5U) % 211U) / 211.0F;
        weights_[index] = 0.05F * std::sin(phase * 6.2831853F);
    }
}

math::Vector EmbeddingTable::encode(
    const std::vector<std::uint32_t>& ids) const {
    math::Vector result(dimension_, 0.0F);
    std::size_t count = 0;
    for (const std::uint32_t id : ids) {
        if (id >= capacity_) {
            continue;
        }
        const std::size_t offset = static_cast<std::size_t>(id) * dimension_;
        for (std::size_t index = 0; index < dimension_; ++index) {
            result[index] += weights_[offset + index];
        }
        ++count;
    }
    if (count == 0U) {
        return result;
    }
    const float inverse = 1.0F / static_cast<float>(count);
    for (float& value : result) {
        value *= inverse;
    }
    return result;
}

void EmbeddingTable::save(std::ostream& output) const {
    write_size(output, dimension_);
    write_size(output, capacity_);
    write_size(output, weights_.size());
    output.write(reinterpret_cast<const char*>(weights_.data()),
                 static_cast<std::streamsize>(weights_.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write embedding table");
    }
}

void EmbeddingTable::load(std::istream& input) {
    if (read_size(input) != dimension_ || read_size(input) != capacity_) {
        throw std::runtime_error("embedding table dimensions do not match");
    }
    const std::size_t size = read_size(input);
    if (size != weights_.size()) {
        throw std::runtime_error("embedding table size does not match");
    }
    input.read(reinterpret_cast<char*>(weights_.data()),
               static_cast<std::streamsize>(weights_.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("embedding table checkpoint is incomplete");
    }
    for (const float value : weights_) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("embedding table contains a non-finite value");
        }
    }
}

} // namespace astrax
