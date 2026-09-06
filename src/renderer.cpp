#include "astrax/renderer.hpp"

#include <iomanip>
#include <sstream>

namespace astrax {

std::string ControlledRenderer::render_text(const std::string& request,
                                            const ModelOutput& output,
                                            const AstraxState& state) const {
    std::ostringstream result;
    result << "Astrax iteration " << state.iteration
           << ". Action " << output.action_id
           << " selected with confidence " << std::fixed << std::setprecision(3)
           << output.confidence << ".";
    if (!state.current_goal.description.empty()) {
        result << " Current goal: " << state.current_goal.description << ".";
    }
    if (!request.empty()) {
        result << " Observation received: " << request;
    }
    return result.str();
}

std::string ControlledRenderer::render_code(const std::string& request,
                                            const ModelOutput& output,
                                            const AstraxState& state) const {
    (void)request;
    std::ostringstream result;
    result << "#include <iostream>\n\n"
           << "int main() {\n"
           << "    constexpr unsigned long long iteration = " << state.iteration << "ULL;\n"
           << "    constexpr unsigned action = " << output.action_id << "U;\n"
           << "    std::cout << \"Astrax action \" << action\n"
           << "              << \" at iteration \" << iteration << '\\\\n';\n"
           << "    return 0;\n"
           << "}\n";
    return result.str();
}

std::string ControlledRenderer::render_state(const AstraxState& state) const {
    std::ostringstream result;
    result << "version=" << state.version
           << " iteration=" << state.iteration
           << " uptime_seconds=" << state.uptime_seconds
           << " memory_count=" << state.memory_count
           << " goal=\"" << state.current_goal.description << "\"";
    return result.str();
}

} // namespace astrax

