#pragma once
#include <vector>

namespace yan_lamma{
struct Batch {
std::vector<int> tokens;
std::vector<int> poss;
// decode 流程
Batch Single(int token, int pos);
// profill 流程
Batch FromTokens(std::vector<int>& tokens, int pos);
};



}// namepace yan_lamma