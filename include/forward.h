#include "include/context.h"
#include "include/kvcache.h"
#include "include/model.h"
#include "include/ops.h"
#include "include/tensor.h"
#include "include/batch.h"
namespace yan_lamma {
// - 查嵌入表 Embedding 映射
// - layer 层级运算
// - Rope 注入信息
// - Residual 残差连接
Tensor ForwardToken(const Model& model, Context& context, int token);
Tensor ForwardBatch(const Model& model, Context& context, const Batch& batch);


}//namespace yan_lamma
