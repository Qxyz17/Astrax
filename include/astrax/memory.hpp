#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "astrax/types.hpp"

namespace astrax {

class MemoryStore {
public:
    MemoryStore(std::size_t vector_dim, std::size_t capacity);

    std::uint64_t write(const std::string& content,
                        const math::Vector& embedding,
                        float salience,
                        std::uint64_t iteration);
    std::vector<MemoryRecord> retrieve(const math::Vector& query,
                                       std::size_t limit,
                                       std::uint64_t iteration);
    void forget_below(float salience);
    void clear() noexcept;

    std::size_t size() const noexcept { return records_.size(); }
    const std::vector<MemoryRecord>& records() const noexcept { return records_; }

private:
    std::size_t vector_dim_;
    std::size_t capacity_;
    std::uint64_t next_id_ = 1;
    std::vector<MemoryRecord> records_;
};

} // namespace astrax

