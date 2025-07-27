from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="hgemm",
    packages=[],
    ext_modules=[
        CUDAExtension(
            name="hgemm",
            sources=[
                "cublas/hgemm.cu",
                "pybind/binds.cpp"
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
