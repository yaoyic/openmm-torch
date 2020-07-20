from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
nn_plugin_header_dir = '@NN_PLUGIN_HEADER_DIR@'
nn_plugin_library_dir = '@NN_PLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
# for python ABI compatibility 
extra_compile_args = ['-std=c++11', '-D_GLIBCXX_USE_CXX11_ABI=0', '-fPIC']
extra_link_args = ['-D_GLIBCXX_USE_CXX11_ABI=0', '-fPIC']

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_openmmtorch',
                      sources=['TorchPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMTorch'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), nn_plugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), nn_plugin_library_dir],
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib')],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='openmmtorch',
      version='1.0',
      py_modules=['openmmtorch'],
      ext_modules=[extension],
     )

