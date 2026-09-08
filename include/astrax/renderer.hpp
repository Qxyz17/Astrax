#pragma once

#include <string>

#include "astrax/types.hpp"

namespace astrax {

class ControlledRenderer {
public:
    std::string render_state(const AstraxState& state) const;
    std::string render_decision(const ModelOutput& output,
                                const AstraxState& state) const;
};

} // namespace astrax
