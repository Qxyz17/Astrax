#include "astrax/memory.hpp"

#include <algorithm>
#include <stdexcept>

#include "astrax/math.hpp"

namespace astrax {

MemoryStore::MemoryStore(std::size_t vector_dim, std::size_t capacity)
    : vector_dim_(vector_dim), capacity_(capacity) {
    if (vector_dim_ == 0 || capacity_ == 0) {
        throw std::invalid_argument("memory dimensions must be positive");
    }
}

std::uint64_t MemoryStore::write(const std::string& content,
                                 const math::Vector& embedding,
                                 float salience,
                                 std::uint64_t iteration) {
    math::require_size(embedding, vector_dim_, "memory embedding");
    if (records_.size() == capacity_) {
        records_.erase(std::min_element(
            records_.begin(), records_.end(), [](const MemoryRecord& left,
                                                 const MemoryRecord& right) {
                return left.salience < right.salience;
            }));
    }
    MemoryRecord record;
    record.id = next_id_++;
    record.content = content;
    record.embedding = embedding;
    record.salience = std::clamp(salience, 0.0F, 1.0F);
    record.created_at = iteration;
    record.last_used = iteration;
    records_.push_back(std::move(record));
    return records_.back().id;
}

std::vector<MemoryRecord> MemoryStore::retrieve(const math::Vector& query,
                                                std::size_t limit,
                                                std::uint64_t iteration) {
    math::require_size(query, vector_dim_, "memory query");
    std::vector<std::pair<float, std::size_t>> ranking;
    ranking.reserve(records_.size());
    for (std::size_t index = 0; index < records_.size(); ++index) {
        const MemoryRecord& record = records_[index];
        const float score = math::cosine(query, record.embedding) *
                            (0.5F + 0.5F * record.salience);
        ranking.emplace_back(score, index);
    }
    std::sort(ranking.begin(), ranking.end(),
              [](const auto& left, const auto& right) {
                  return left.first > right.first;
              });

    std::vector<MemoryRecord> result;
    for (std::size_t rank = 0; rank < ranking.size() && rank < limit; ++rank) {
        MemoryRecord& record = records_[ranking[rank].second];
        record.last_used = iteration;
        ++record.use_count;
        result.push_back(record);
    }
    return result;
}

void MemoryStore::forget_below(float salience) {
    records_.erase(
        std::remove_if(records_.begin(), records_.end(),
                       [salience](const MemoryRecord& record) {
                           return record.salience < salience;
                       }),
        records_.end());
}

void MemoryStore::clear() noexcept {
    records_.clear();
}

} // namespace astrax

