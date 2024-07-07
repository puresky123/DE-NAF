#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <algorithm>
#include <stdexcept>

#include <cstdio>

// 检查张量是否在CUDA上
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
// 检查张量是否内存连续
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
// 检查张量类型是否是整数
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
// 检查张量类型是否是浮点类型(包括float、half、double)等
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


// requires CUDA >= 10 and ARCH >= 70
// this is very slow compared to float or __half2, do not use!
// 自定义的CUDA原子加法函数，用于在GPU上对at::Half（半精度浮点数）类型进行原子加操作，但是这种方式计算很慢，慎重使用
static inline  __device__ at::Half atomicAdd(at::Half *address, at::Half val) {
  return atomicAdd(reinterpret_cast<__half*>(address), val);
}

// 这是一个模板函数，用于将输入的(val + divisor - 1)除以divisor并向上取整，得到结果。该函数可以在主机端（host）和设备端（device，即GPU）上使用，
// 使用__host__ __device__修饰符表示这个函数既可以在CPU上编译执行，也可以在GPU上作为设备函数执行。
template <typename T>
static inline __host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

// 这是一个模板函数，用于将高维坐标pos_grid（维度为D）映射为一个哈希值。该函数用于计算D维坐标的哈希值，将高维坐标映射为一个较低维度的索引。
// 函数中使用了一系列常量素数来进行哈希运算，以在高维空间中生成一个均匀分布的哈希索引。该函数被声明为设备函数，只能在CUDA设备上执行。
template <uint32_t D>
__device__ uint32_t fast_hash(const uint32_t pos_grid[D]) {
	static_assert(D <= 7, "fast_hash can only hash up to 7 dimensions.");

	// While 1 is technically not a good prime for hashing (or a prime at all), it helps memory coherence
	// and is sufficient for our use case of obtaining a uniformly colliding index from high-dimensional
	// coordinates.
	constexpr uint32_t primes[7] = { 1, 19349663, 83492791, 25165843, 6291469, 12582917, 3145739 };

	uint32_t result = 0;
	#pragma unroll
	for (uint32_t i = 0; i < D; ++i) {
		result ^= pos_grid[i] * primes[i];
	}

	return result;
}

// 计算一个高维坐标（pos_grid）在哈希表（Grid哈希表）中的索引, 这个函数在CUDA设备上执行
template <uint32_t D, uint32_t C>
__device__ uint32_t get_grid_index(const uint32_t ch, const uint32_t hashmap_size, const uint32_t resolution, const uint32_t pos_grid[D]) {
	uint32_t stride = 1;
	uint32_t index = 0;

	#pragma unroll
    for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
        //printf("get_grid_index d=%d, pos_grid[d]=%d, stride=%d, reso=%d\n", d, pos_grid[d], stride, resolution);
        index += pos_grid[d] * stride;
        stride *= (resolution + 1);
    }
    // 检查计算出的步长是否超过了哈希表的大小。如果步长超过了哈希表大小，则说明高维坐标在哈希表中不是有效的索引，此时会调用 fast_hash<D>(pos_grid) 函数对高维坐标进行哈希运算，
    // 将其映射到一个较低维度的索引。这样做的目的是为了确保高维坐标在哈希表中能够得到一个较均匀的哈希索引。
    if (stride > hashmap_size) {
        //printf("hash because %d > %d\n", stride, hashmap_size);
        index = fast_hash<D>(pos_grid);
        //printf("hashed (%d, %d) = %d to %d in %d\n", pos_grid[0], pos_grid[1], pos_grid[0] + resolution * pos_grid[1], index % hashmap_size, hashmap_size);
    }

	return (index % hashmap_size) * C + ch;
}

// 这段代码是一个CUDA核函数（Kernel Function），用于在CUDA设备上进行高维空间的格子哈希编码（Grid Hashing）操作。
// 核函数是在GPU上并行执行的函数，每个线程块（block）内的线程可以同时处理多个数据。
template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid(
    const scalar_t * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ outputs, 
    uint32_t B, uint32_t L, uint32_t H,
    const bool calc_grad_inputs, 
    scalar_t * __restrict__ dy_dx
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x; // 获取当前线程在块内的索引和块索引，计算当前线程的总索引 b
    
    if (b >= B) return; // 如果当前线程索引超过了输入张量 B 的大小，则返回，不执行任何操作

    const uint32_t level = blockIdx.y; // 获取当前块在 grid 中的索引 level
    
    grid += (uint32_t)offsets[level] * C; // 定位 grid 张量在当前块的位置
    inputs += b * D; // 定位 inputs 张量在当前线程的位置
    outputs += level * B * C + b * C; // 定位 outputs 张量在当前块和当前线程的位置

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level]; // 计算当前层级 hashmap 的大小
    const float scale = exp2f(level) * H - 1.0f; // 计算当前层级的缩放比例
    const uint32_t resolution = (uint32_t)ceil(scale) + 1; // 计算当前层级的分辨率
    
   // 计算输入坐标pos[D]在grid上对应的格点pos_grid[D]
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = (float)inputs[d] * scale + 0.5f; // 计算当前维度上的坐标值 pos
        pos_grid[d] = floorf(pos[d]); // 计算当前维度上的坐标格点 pos_grid
        pos[d] -= (float)pos_grid[d];
    }

    //printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1], pos_grid[0], pos_grid[1]);

    // 插值操作
    float results[C] = {0}; // 用于暂存结果的寄存器数组

    // 循环遍历所有可能的索引组合 (1 << D) 进行插值
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1; // 初始化插值权重 w 为 1
        uint32_t pos_grid_local[D];

        // 遍历每个维度，根据 idx 来计算当前的插值权重 w 和 pos_grid_local
        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local); // 计算在 grid 张量中的索引位置

        //  在寄存器数组 results 中累加插值结果 writing to register (fast)
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += w * grid[index + ch];
        }

        //printf("[b=%d, l=%d] int %d, idx %d, w %f, val %f\n", b, level, idx, index, w, grid[index]);
    }    

    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        outputs[ch] = results[ch]; 
    }

    // prepare dy_dx for calc_grad_inputs
    // differentiable (soft) indexing: https://discuss.pytorch.org/t/differentiable-indexing/17647/9
    if (calc_grad_inputs) {
        // 将 dy_dx 定位到当前线程和当前层级的位置
        dy_dx += b * D * L * C + level * D * C; // B L D C

        // 循环遍历所有可能的索引组合 (1 << (D - 1)) 进行梯度计算
        #pragma unroll
        for (uint32_t gd = 0; gd < D; gd++) {

            float results_grad[C] = {0}; // temp

            // 遍历每个维度计算梯度权重 w 和 pos_grid_local
            #pragma unroll
            for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
                //float w = scale;
                float w = 1;
                uint32_t pos_grid_local[D];

                #pragma unroll
                for (uint32_t nd = 0; nd < D - 1; nd++) {
                    const uint32_t d = nd > gd ? nd + 1 : nd;

                    if ((idx & (1 << nd)) == 0) {
                        w *= 1 - pos[d];
                        pos_grid_local[d] = pos_grid[d];
                    } else {
                        w *= pos[d];
                        pos_grid_local[d] = pos_grid[d] + 1;
                    }
                }
                // 计算在 grid 张量中的左右两个索引位置
                pos_grid_local[gd] = pos_grid[gd];
                uint32_t index_left = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
                pos_grid_local[gd] = pos_grid[gd] + 1;
                uint32_t index_right = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);

                // 计算梯度并在 results_grad 中累加
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    results_grad[ch] += w * (grid[index_right + ch] - grid[index_left + ch]);
                }
            }
            // 将结果写入 dy_dx 张量中
            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                dy_dx[gd * C + ch] = results_grad[ch];
            }
        }
    }
}


template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ grad_grid, 
    uint32_t B, uint32_t L, uint32_t H
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
	if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    // locate
    grad_grid += offsets[level] * C;
    inputs += b * D;
    grad += b * L * C + level * C + ch;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;

    // 计算输入坐标pos[D]在grid上对应的格点pos_grid[D]
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = (float)inputs[d] * scale + 0.5f;
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    // 插值计算
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll // 用于循环展开的指令。它告诉编译器在编译阶段对指定的循环进行展开，以优化计算性能
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(ch, hashmap_size, resolution, pos_grid_local);

        // 使用原子操作（atomicAdd）将梯度写入到输出张量 grad_grid 中，实现并行计算
        // 当通道数 N_C 是格子特征向量维度的整数倍且数据类型是 half 时，使用 __half2 优化计算
        // 否则，逐个通道写入梯度
        // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
        // TODO: use float which is better than __half, if N_C % 2 != 0
        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(grad[c] * w), (__half)(grad[c + 1] * w)};
                atomicAdd((__half2*)&grad_grid[index + c], v);
            }
        // float, or __half when N_C % 2 != 0
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad_grid[index + c], w * grad[c]);
            }
        }
    }    
}


template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ dy_dx,  
    scalar_t * __restrict__ grad_inputs, 
    uint32_t B, uint32_t L
) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x; // 计算当前线程的索引
    if (t >= B * D) return; // 如果当前索引超出输入数据的大小，则退出

    // 计算当前线程所在的 batch 和维度
    const uint32_t b = t / D;
    const uint32_t d = t - b * D;

    // 根据当前 batch 和维度，定位到 grad 和 dy_dx 对应的位置
    grad += b * L * C;
    dy_dx += b * L * D * C;
    
    # pragma unroll
    for (int l = 0; l < L; l++) { // 循环遍历每个分辨率等级 L
        # pragma unroll
        for (int ch = 0; ch < C; ch++) { // 循环遍历每个通道 C
            // 根据公式 grad_inputs[t] += grad[l * C + ch] * dy_dx[l * D * C + d * C + ch] 更新 grad_inputs[t] 的值
            grad_inputs[t] += grad[l * C + ch] * dy_dx[l * D * C + d * C + ch];
        }
    }
}

// kernel_grid_wrapper 函数，用于调用 kernel_grid 核函数的包装函数
template <typename scalar_t, uint32_t D>
void kernel_grid_wrapper(const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx) {
    static constexpr uint32_t N_THREAD = 512;
	const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grid<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, H, calc_grad_inputs, dy_dx); break;
        case 2: kernel_grid<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, H, calc_grad_inputs, dy_dx); break;
        case 4: kernel_grid<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, H, calc_grad_inputs, dy_dx); break;
        case 8: kernel_grid<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, H, calc_grad_inputs, dy_dx); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

// hash_encode_forward_cuda 函数，用于调用 kernel_grid_wrapper 核函数的包装函数
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [L, B, C], float (L first, so only one level of hashmap needs to fit into cache at a time.)
// H: base resolution
template <typename scalar_t>
void hash_encode_forward_cuda(const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx) {
    switch (D) {
        case 2: kernel_grid_wrapper<scalar_t, 2>(inputs, embeddings, offsets, outputs, B, C, L, H, calc_grad_inputs, dy_dx); break;
        case 3: kernel_grid_wrapper<scalar_t, 3>(inputs, embeddings, offsets, outputs, B, C, L, H, calc_grad_inputs, dy_dx); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
    
}

template <typename scalar_t, uint32_t D>
void kernel_grid_backward_wrapper(const scalar_t *grad, const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx, scalar_t *grad_inputs) {
    static constexpr uint32_t N_THREAD = 256;
	const uint32_t N_C = std::min(2u, C); // n_features_per_thread
	const dim3 blocks_hashgrid = { div_round_up(B * C / N_C, N_THREAD), L, 1 };
    switch (C) {
        case 1: 
            kernel_grid_backward<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, H); 
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 2: 
            kernel_grid_backward<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, H);
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 4: 
            kernel_grid_backward<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, H);
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 8: 
            kernel_grid_backward<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, H);
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


// grad: [B, L * C], float
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// grad_embeddings: [sO, C]
// H: base resolution
template <typename scalar_t>
void hash_encode_backward_cuda(const scalar_t *grad, const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx, scalar_t *grad_inputs) {
    switch (D) {
        case 2: kernel_grid_backward_wrapper<scalar_t, 2>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, H, calc_grad_inputs, dy_dx, grad_inputs); break;
        case 3: kernel_grid_backward_wrapper<scalar_t, 3>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, H, calc_grad_inputs, dy_dx, grad_inputs); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}



void hash_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(outputs);
    CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(outputs);
    CHECK_IS_FLOATING(dy_dx);
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND_HALF 宏，对输入的 inputs 张量的数据类型进行分发
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inputs.type(), "hash_encode_forward", ([&] {
        // 调用 hash_encode_forward_cuda 包装函数，传递对应的数据指针和参数
        // scalar_t 为模板参数，表示当前数据类型的实际类型
        hash_encode_forward_cuda<scalar_t>(inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), outputs.data_ptr<scalar_t>(), B, D, C, L, H, calc_grad_inputs, dy_dx.data_ptr<scalar_t>());
    }));
}

void hash_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(grad_embeddings);
    CHECK_CUDA(dy_dx);
    CHECK_CUDA(grad_inputs);
    
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grad_embeddings);
    CHECK_CONTIGUOUS(dy_dx);
    CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(grad_embeddings);
    CHECK_IS_FLOATING(dy_dx);
    CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.type(), "hash_encode_backward", ([&] {
        hash_encode_backward_cuda<scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<scalar_t>(), B, D, C, L, H, calc_grad_inputs, dy_dx.data_ptr<scalar_t>(), grad_inputs.data_ptr<scalar_t>());
    }));
    
}
