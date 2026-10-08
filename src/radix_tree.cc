#include "include/radix_tree.h"

#include <utility>
#include <algorithm>
namespace yan_lamma {
RadixTree::RadixTree(RadixTree&& other) noexcept
    : root_(std::move(other.root_)), terminal_count(other.terminal_count) {
    other.Clear();
}

bool RadixTree:: Empty() {
    return terminal_count == 0;
}
void RadixTree::Clear() {
    root_.children.clear();
    root_.terminal = false;
    root_.key.clear();
    terminal_count = 0;

}
size_t RadixTree:: CommonPrefixLength(const std::vector<int>& a, size_t offset, const std::vector<int>& b) {
    size_t pos = 0;
    while(offset + pos < a.size() && pos < b.size() && a[offset + pos] == b[pos]) {
        pos++;
    }
    return pos;
}

void RadixTree::InsertInfo(Node* parent, std::vector<int>& tokens) {
    auto it = parent->children.find(tokens[0]);
    if (it == parent->children.end()) {
        auto ch = std::make_unique<Node>();
        ch->key = tokens;
        ch-> terminal =true;
        const int first_token = ch->key[0];
        parent->children[first_token] = std::move(ch);
        terminal_count++;
        return;
    }
    std::unique_ptr<Node>& child = it->second;
    size_t common =  CommonPrefixLength(tokens, 0, child->key);
    if (common == child->key.size()) {
        if (common == tokens.size()) {
            if (!child->terminal) {
                 child->terminal = true;
                 terminal_count++;
            }
            return;
        }
         std::vector<int> temp(tokens.begin() + static_cast<long>(common), tokens.end());
         InsertInfo(child.get(), temp);
         return;
    }
    auto split = std::make_unique<Node>();
    split->key.assign(tokens.begin(), tokens.begin() + common);
    child->key.erase(child->key.begin(), child->key.begin() + common);
    int old_child_first = child->key[0];
    split->children[old_child_first] = std::move(child);
    if (common == tokens.size()) {
        split->terminal = true;
        terminal_count++;
    } else {
        auto new_child = std::make_unique<Node>();
        new_child->key.assign(tokens.begin() + static_cast<long>(common), tokens.end());
        new_child->terminal = true;
        terminal_count++;
        split->children[new_child->key[0]] = std::move(new_child);
    }
   it->second = std::move(split);
}

size_t RadixTree::LongestPrefis(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        return 0;
    }
    Node* temp = &root_;
    size_t pos = 0;
    while (pos < tokens.size()) {
        auto child_it = temp->children.find(tokens[pos]);
        if (child_it ==  temp->children.end()) {
            return pos;
        }
         Node* child = child_it->second.get();
        size_t common =  CommonPrefixLength(tokens, pos, child->key);
        pos += common;
        if (common < child->key.size()) {
            return pos;
        }
        temp = child;
    }
    return pos;
}
void RadixTree::Insert(std::vector<int>& tokens) {
    if (tokens.empty()) {
        return;
    }
    InsertInfo(&root_, tokens);
}
}// namespace yan_lamma