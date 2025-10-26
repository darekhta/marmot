#pragma once

#include "marmot/graph/graph.hpp"

struct marmot_graph {
    marmot::graph::Graph inner;
    void (*external_cleanup)(void *){nullptr};
    void *external_state{nullptr};
};
