import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd # 导入PyTorch的amp模块，用于半精度训练

from .backend import _backend

# 定义_hash_encode类用于实现hash_encode的前向传播和反向传播
class _hash_encode(Function):
    @staticmethod
    # @custom_fwd # 另一种前向传播时的amp自定义装饰器，不使用半精度浮点数计算
    @custom_fwd(cast_inputs=torch.half) # 使用amp自定义装饰器，指定前向传播时使用半精度浮点数计算
    def forward(ctx, inputs, embeddings, offsets, base_resolution, calc_grad_inputs=False):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        # 如果input存储不连续，inputs.contiguous()会拷贝一份inputs，在后面对拷贝的inputs进行修改时不会影响作为输入的inputs, embeddings和offsets同理
        # 确保输入张量 inputs 是连续存储的，以提高后续计算的效率
        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous().to(inputs.device)

        B, D = inputs.shape # B = batch size, D = coord dim
        L = offsets.shape[0] - 1 # L = num_levels
        C = embeddings.shape[1] # C = level_dim ，即embedding dim for each level
        H = base_resolution # 最低分辨率，论文中的N_min

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.zeros(L, B, C, device=inputs.device, dtype=inputs.dtype) # 初始化outputs: [L, B, C]

        if calc_grad_inputs:
            dy_dx = torch.zeros(B, L * D * C, device=inputs.device, dtype=inputs.dtype) # 如果需要计算输入梯度，则初始化一个大小为[B, L * D * C]的dy_dx来存储梯度信息
        else:
            dy_dx = torch.zeros(1, device=inputs.device, dtype=inputs.dtype) # 否则初始化一个标量值，仅用于占位
        #
        _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, H, calc_grad_inputs, dy_dx)

        outputs = outputs.permute(1, 0, 2).reshape(B, L * C) # [L, B, C] ——> [B, L, C] ——> [B, L * C]

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx) # 将输入的inputs、embeddings、offsets等数据按顺序保存在内存中，以备反向传播时使用
        ctx.dims = [B, D, C, L, H] # 保存输入的一些维度信息，供反向传播时使用
        ctx.calc_grad_inputs = calc_grad_inputs # 保存计算梯度的标志位

        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous() # 将输入的梯度grad进行内存连续化; grad: [B, L * C]

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors # 从保存的ctx对象中获取前向传播时保存的inputs、embeddings、offsets、dy_dx等数据
        B, D, C, L, H = ctx.dims # 获取前向传播时保存的维度信息
        calc_grad_inputs = ctx.calc_grad_inputs # 获取计算梯度的标志位

        grad_embeddings = torch.zeros_like(embeddings) # 初始化一个与embeddings维度相同的 grad_embeddings 用于存储embeddings的梯度

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs) # 如果计算输入梯度，则初始化一个与inputs维度相同的 grad_inputs 用于存储inputs的梯度
        else:
            grad_inputs = torch.zeros(1, device=inputs.device, dtype=inputs.dtype) # 否则初始化一个标量值，仅用于占位
        #
        _backend.hash_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, H, calc_grad_inputs, dy_dx, grad_inputs)

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None # 返回计算得到的输入梯度grad_inputs和embeddings梯度grad_embeddings
        else:
            return None, grad_embeddings, None, None, None # 返回计算得到的embeddings梯度grad_embeddings

# 使用自定义Function类_hash_encode创建一个新的torch.autograd.Function，并命名为hash_encode
# 在PyTorch中，自定义的Function类需要通过 .apply 方法进行注册，以便在后续的计算中能够被正确识别和调用。
# 即通过 .apply 以便PyTorch识别并将其合并到向前和向后传递的计算图中。通过注册自定义函数，PyTorch意识到它的存在，并可以将其包含在其自动梯度系统中，该系统负责在神经网络的反向传递期间自动微分和计算梯度
hash_encode = _hash_encode.apply

# 定义HashEncoder类，继承自nn.Module类，用于实现 hash编码
class HashEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19):
        super().__init__()

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level，论文中的F
        self.log2_hashmap_size = log2_hashmap_size # 论文中T的最高次数
        self.base_resolution = base_resolution # 最低分辨率, 论文中的N_min
        self.output_dim = num_levels * level_dim # 每个分辨率等级所编码的特征向量的维度都为level_dim，因此输出维度为 num_levels * level_dim

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

        # 分配参数
        self.offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size # 设置每级最大参数量（为防止随着分辨率等级的提升所需的参数量过大），论文中的T
        # 计算各级分辨率的具体数值
        for i in range(num_levels):
            resolution = base_resolution * 2 ** i # 每个分辨率等级的分辨率比前一级提高一倍
            # resolution大小的分辨率可以分出(resolution + 1)个点，对三维空间来说，共有(resolution + 1) ** 3个点，这些点存储了包含该点在内邻近区域的信息
            params_in_level = min(self.max_params, (resolution + 1) ** input_dim) # 每级的参数量取 该级对应分辨率体素网格数 与 最大参数量 中的最小值
            # params_in_level = int(params_in_level / 8) * 8 # make divisible
            self.offsets.append(offset)
            offset += params_in_level
        self.offsets.append(offset)
        self.offsets = torch.from_numpy(np.array(self.offsets, dtype=np.int32)) # 将offsets从Python的NumPy数组转换为PyTorch张量
        self.n_params = self.offsets[-1] * level_dim # 计算总的参数个数，总参数量 = 所有分辨率等级所需点的数量之和 * 每个点表示的特征向量的维度

        # 定义可训练的参数embeddings: [offset, level_dim]; embeddings保存了各级hash表中的全部可学习参数
        self.embeddings = nn.Parameter(torch.zeros(offset, level_dim))
        # 初始化 self.embeddings 中的参数
        self.reset_parameters()

    # hash表参数初始化函数
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std) # 用(-std, std)之间的均匀分布来初始化embeddings，即hash表中的参数

    # 指定一个对象的文本表述，即创建一个对象时，会在控制台显示出对该对象的文本描述
    def __repr__(self):
        return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} H={self.base_resolution} params={self.embeddings.shape}"
    
    def forward(self, inputs, size=1):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]

        # 检查 inputs 是否在 [-size, size] 内
        if inputs.min().item() < -size or inputs.max().item() > size:
            raise ValueError(f'HashGrid encoder: inputs range [{inputs.min().item()}, {inputs.max().item()}] not in [{-size}, {size}]!')
        # 将输入从[-size, size]映射到[0, 1] (map to [0, 1])
        inputs = (inputs + size) / (2 * size)

        # max-min 归一化
        # min_vals, _ = torch.min(inputs, dim=-2, keepdim=True)
        # max_vals, _ = torch.max(inputs, dim=-2, keepdim=True)
        # inputs = (inputs - min_vals) / (max_vals - min_vals)

        # 测试代码，用来查看inputs
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())
        """
            inputs的最后一维是坐标的维度，前面几维未具体设置，视具体情况而定。先用 prefix_shape 保存 inputs 除最后一维外前面几维的shape，
            之后将inputs的除最后一维外的前面几维展平（prefix_shape在后面起复原作用），并输入hash_encode编码获得outputs，最后将outputs除
            最后一维外的前面几维通过之前保存的prefix_shape来还原
        """
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)
        # inputs.requires_grad 表示在训练中保留 inputs的梯度用于后续更新
        outputs = hash_encode(inputs, self.embeddings.cuda(), self.offsets, self.base_resolution, inputs.requires_grad)
        outputs = outputs.view(prefix_shape + [self.output_dim])
        # 测试代码，用来查看outputs
        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())
        return outputs