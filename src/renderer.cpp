#include "astrax/renderer.hpp"

#include <sstream>

namespace astrax {

std::string ControlledRenderer::render_state(const AstraxState& state) const {
    std::ostringstream result;
    result << "iteration=" << state.iteration
           << " uptime_seconds=" << state.uptime_seconds
           << " memory_count=" << state.memory_count
           << " goal=\"" << state.current_goal.description << "\"";
    return result.str();
}

std::string ControlledRenderer::render_decision(const ModelOutput& output,
                                                const AstraxState& state) const {
    std::ostringstream result;
    result << "decision action=" << output.action_id
           << " engine_action=" << output.engine_action_id
           << " action_value=" << output.action_value
           << " confidence=" << output.confidence
           << " intrinsic_reward=" << output.intrinsic_reward
           << " iteration=" << state.iteration;
    return result.str();
}

} // namespace astrax
