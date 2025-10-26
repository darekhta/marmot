#pragma once

#include "marmot/error.h"

#include <memory>
#include <mutex>

namespace marmot::inference {
class Model;
class Session;
class Llm;
class LlmEngine;
class ServingEngine;
} // namespace marmot::inference

struct MarmotModelHandle {
    std::shared_ptr<marmot::inference::Model> impl;
    mutable std::mutex last_error_mutex{};
    mutable marmot_error_info_t last_error{};
};

struct MarmotSessionHandle {
    std::unique_ptr<marmot::inference::Session> impl;
    mutable std::mutex last_error_mutex{};
    mutable marmot_error_info_t last_error{};
};

struct MarmotLlmHandle {
    std::unique_ptr<marmot::inference::Llm> impl;
    mutable std::mutex last_error_mutex{};
    mutable marmot_error_info_t last_error{};
};

struct MarmotLlmEngineHandle {
    std::unique_ptr<marmot::inference::LlmEngine> impl;
    mutable std::mutex last_error_mutex{};
    mutable marmot_error_info_t last_error{};
};

struct MarmotServingEngineHandle {
    std::unique_ptr<marmot::inference::ServingEngine> impl;
    mutable std::mutex last_error_mutex{};
    mutable marmot_error_info_t last_error{};
};
