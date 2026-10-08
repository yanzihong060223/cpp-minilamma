#pragma once
#include <vector>
#include <map>
#include <memory>
namespace yan_lamma {
class RadixTree {
public:
RadixTree() = default;
~RadixTree() = default;
void Clear();
bool Empty();
RadixTree (const RadixTree&) = delete;
RadixTree& operator= (const RadixTree&) = delete;
RadixTree (RadixTree && other) noexcept;
void Insert(std::vector<int>& tokens);
size_t LongestPrefis(const std::vector<int>& tokens);
private:
struct Node {
    // 节点保存的整段token
    std::vector<int> key;
    //是否插入过
    bool terminal = false;
    //查找token的第一个值 便于定位
    std::map<int, std::unique_ptr<Node>> children;
};
//得到公共前缀长度
size_t CommonPrefixLength(const std::vector<int>& a, size_t offset, const  std::vector<int>& b);
void InsertInfo(Node* parent, std::vector<int>& tokens);
Node root_; //根节点
size_t terminal_count = 0;




};
} //namespace yan_lamma