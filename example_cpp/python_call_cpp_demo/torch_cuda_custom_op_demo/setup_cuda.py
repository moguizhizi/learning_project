from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="my_cuda_ops_lib",
    ext_modules=[
        CUDAExtension(
            name="my_cuda_ops_lib",
            sources=["cuda_custom_op.cu"],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False),
    },
)
