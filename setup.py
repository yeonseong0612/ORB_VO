from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='orb_cuda',
    ext_modules=[
        CUDAExtension(
            name='orb_cuda',
            sources=[
                'src/orb/orb_wrapper.cpp',
                'src/orb/torch_bindings.cpp',
                'src/orb/orb.cpp',
                'src/orb/orbd.cu',
                'src/orb/warmup.cu',
            ],
            include_dirs=['.'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)