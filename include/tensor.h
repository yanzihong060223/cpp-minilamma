#pragma once
#include <cstddef>
#include <vector>
#include <string>
namespace yan_lamma {
struct Tensor {
std::vector<float> data;
std::vector<int> shape;
Tensor() = default;
explicit Tensor(const std::vector<int>& data_shape, float default_data = 0.0f);
//获得数据大小
std::size_t GetDataSize() const {return data.size();}
std::size_t GetNumDims() const {return shape.size();}
//提供安全的at获取元素/修改元素
float At1(int i) const ; //一维
float& At1(int i);
float At2(int i, int j) const ;
float& At2 (int i, int j);
float At3(int i, int j, int k) const;
float& At3(int i, int j, int k);
float At4(int i, int j, int k, int m) const;
float& At4(int i, int j, int k, int m);
float* RowPtr(int row) ; //得到某行的原始指针
const float* RowPtr(int row) const; 
// 多维转一维
std::size_t SwitchIdex(const std::vector<int>& indices) const;
//直接访问
float& operator[] (std::size_t i) {return data[i];}
float operator[] (std::size_t i)  const {return data[i];}
//检查失败 抛出异常
void AssertShape(const std::vector<int>& expected, const char* call);
Tensor Reshape(const std::vector<int>& new_shape, const char* call);
//调试 判断形状
std::string ShapeToString() const;
void Print(const std::string& name = "", bool print_data = true) const;
//创建一维到四维的tensor
Tensor MakeOneD(int i, float default_data = 0.0f);
Tensor MakeSecondD(int i, int j, float default_data = 0.0f);
Tensor MakeThirdD(int i, int j, int k, float default_data = 0.0f);
Tensor MakeFouthD(int i, int j, int k, int m, float default_data = 0.0f);
};
} //namespace yan_lamma
