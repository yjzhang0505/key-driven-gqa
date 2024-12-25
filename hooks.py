import torch

def register_hooks(model, hook_fn):
    """
    将钩子注册到每个 Attention 模块（self.att）
    """
    hooks = []
    
    # 遍历 Vision Transformer 中的每个 block
    for i, block in enumerate(model.blocks):
        # 注册到每个 Block 中的 Attention 模块（即 self.att）
        hook = block.attn.v.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    return hooks

def hook_fn(model, input, output):
    """
    钩子函数，用于提取 Attention 输出，并计算海森矩阵
    """
    # 输出的形状
    print("Attention output shape:", output.shape)
    
    # 获取当前模型的梯度
    output = output.sum()  # 将输出标量化为标量，便于计算梯度
    
    # 清零现有的梯度
    model.zero_grad()

    # 计算一阶梯度
    output.backward(retain_graph=True)

    # 获取模型参数的梯度
    grads = [param.grad for param in model.parameters()]

    # 计算海森矩阵
    hessian = compute_hessian(model, grads)
    
    # 打印或者保存海森矩阵
    print("Hessian matrix:", hessian)

def compute_hessian(model, grads):
    """
    计算海森矩阵
    """
    hessian = {}
    
    # 对每个参数计算二阶导数
    for i, param in enumerate(model.parameters()):
        if param.grad is not None:
            param_shape = param.grad.shape
            hessian_matrix = torch.zeros(param_shape, param_shape)
            # 对每个参数元素计算二阶导数
            for j in range(param.numel()):
                grad_j = grads[i].view(-1)[j]
                grad_2nd = torch.autograd.grad(grad_j, param, create_graph=True)[0]
                hessian_matrix[j] = grad_2nd.view(-1)[j]
            hessian[param] = hessian_matrix
    return hessian
