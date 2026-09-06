#pragma once

#include <string>

#include "astrax/types.hpp"

namespace astrax {

class ControlledRenderer {
public:
    std::string render_text(const std::string& request,
                            const ModelOutput& output,
                            const AstraxState& state) const;
    std::string render_code(const std::string& request,
                            const ModelOutput& output,
                            const AstraxState& state) const;
    std::string render_state(const AstraxState& state) const;
};

} // namespace astrax
