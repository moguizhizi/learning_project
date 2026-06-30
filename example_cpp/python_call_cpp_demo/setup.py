from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="cpp_ops_demo",
    ext_modules=[
        CppExtension(
            name="cpp_ops_demo",
            sources=["cpp_ops.cpp"],
            extra_compile_args=["-O2"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False),
    },
)
