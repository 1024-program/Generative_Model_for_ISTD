import torch

# 创建一个三维张量
tensor_3d = torch.tensor([[[1, 0, 2], [3, 0, 4], [5, 0, 6]],
                          [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                          [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32)


# 获取非零元素的索引
nonzero_indices = [torch.nonzero(tensor_matrix) for tensor_matrix in tensor_3d]

# 获取非零元素的值
var_list = []
for bs in range(tensor_3d.size(0)):
    nonzero_indice = nonzero_indices[bs]
    tensor_matrix = tensor_3d[bs]
    nonzero_values = tensor_matrix[nonzero_indice[:, 0], nonzero_indice[:, 1]]

    # 计算非零元素的方差
    variance = torch.var(nonzero_values)
    var_list

print("非零元素的方差为：", variance)

