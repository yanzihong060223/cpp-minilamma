#pragma once
#include "include/model.h"

#include <string>

namespace yan_lamma {
    Model ModelLoad(const std::string& gguf);
    void VailedShape(Model& model);
}