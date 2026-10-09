#include "include/batch.h"

#include <utility>

namespace yan_lamma {
Batch Batch::Single(int token, int pos) {
    Batch batch;
    batch.tokens.push_back(token);
    batch.poss.push_back(pos);
    return batch;
}

Batch Batch::FromTokens(std::vector<int>& tokens, int pos) {
    Batch batch;
    batch.tokens = tokens;
    batch.poss.reserve(batch.tokens.size());
    for (std::size_t i = 0; i < batch.tokens.size(); ++i) {
        batch.poss.push_back(pos + static_cast<int>(i));
    }
    return batch;
}







}//namespace yan_lamma