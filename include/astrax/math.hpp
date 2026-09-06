#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace astrax::math {

using Vector = std::vector<float>;

inline void require_size(const Vector& value, std::size_t expected, const char* name) {
    if (value.size() != expected) {
        throw std::invalid_argument(std::string(name) + " has an unexpected size");
    }
}

inline float dot(const Vector& left, const Vector& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("vector dimensions do not match");
    }
    float result = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result += left[index] * right[index];
    }
    return result;
}

inline float norm(const Vector& value) {
    return std::sqrt(std::max(0.0F, dot(value, value)));
}

inline float cosine(const Vector& left, const Vector& right) {
    const float denominator = norm(left) * norm(right);
    return denominator > 1.0e-6F ? dot(left, right) / denominator : 0.0F;
}

inline Vector add(const Vector& left, const Vector& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("vector dimensions do not match");
    }
    Vector result(left.size(), 0.0F);
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = left[index] + right[index];
    }
    return result;
}

inline Vector subtract(const Vector& left, const Vector& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("vector dimensions do not match");
    }
    Vector result(left.size(), 0.0F);
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = left[index] - right[index];
    }
    return result;
}

inline Vector scale(const Vector& value, float factor) {
    Vector result = value;
    for (float& item : result) {
        item *= factor;
    }
    return result;
}

inline Vector concatenate(std::initializer_list<const Vector*> values) {
    Vector result;
    for (const Vector* value : values) {
        result.insert(result.end(), value->begin(), value->end());
    }
    return result;
}

inline Vector zeros(std::size_t size) {
    return Vector(size, 0.0F);
}

inline Vector one_hot(std::size_t index, std::size_t count) {
    if (index >= count) {
        throw std::invalid_argument("one-hot index is out of range");
    }
    Vector result(count, 0.0F);
    result[index] = 1.0F;
    return result;
}

inline std::size_t argmax(const Vector& values) {
    if (values.empty()) {
        throw std::invalid_argument("argmax requires a non-empty vector");
    }
    return static_cast<std::size_t>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

} // namespace astrax::math

