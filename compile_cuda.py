from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Define the C++ and CUDA source files
csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hyper_ssm', 'csrc')
sources = [
    os.path.join(csrc_dir, 'riemannian.cpp'),
    os.path.join(csrc_dir, 'riemannian_kernel.cu')
]

setup(
    name='hyper_ssm_cuda',
    ext_modules=[
        CUDAExtension(
            name='hyper_ssm_cuda',
            sources=sources,
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-allow-unsupported-compiler']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
