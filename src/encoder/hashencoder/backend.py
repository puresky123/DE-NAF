import os
from torch.utils.cpp_extension import load

# 获取当前文件所在的文件路径
_src_path = os.path.dirname(os.path.abspath(__file__))

# 使用load函数加载了一个名为_hash_encoder的C++扩展模块。该模块由给定的 C++ 和 CUDA源文件(hashencoder.cu和bindings.cpp)编译而成
_backend = load(name='_hash_encoder',
                extra_cflags=['-O3'], # '-std=c++17' # 将CPU编译的优化级别设置为3
                extra_cuda_cflags=[
                    '-O3',  # 将CUDA编译的优化级别设置为3
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', # undefine flags, necessary!
                ], # '-arch=sm_70' # 为了启用半精度支持，取消了先前在源代码中可能定义的一些CUDA标志
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'hashencoder.cu',
                    'bindings.cpp',
                ]], # CUDA source file: hashencoder.cu; C++ source file: bindings.cpp
                )

# Specify the symbol names that will be imported when someone imports this module
__all__ = ['_backend']