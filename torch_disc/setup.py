# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function
import re
import os
import glob
import inspect
import multiprocessing
import multiprocessing.pool

from setuptools import setup, find_packages, distutils
from torch.utils.cpp_extension import BuildExtension, CppExtension

base_dir = os.path.dirname(os.path.abspath(__file__))

# building with LTC source code
client_files = [
    'pytorch/lazy_tensor_core/third_party/computation_client/env_vars.cc',
    'pytorch/lazy_tensor_core/third_party/computation_client/metrics_reader.cc',
    'pytorch/lazy_tensor_core/third_party/computation_client/sys_util.cc',
    'pytorch/lazy_tensor_core/third_party/computation_client/triggered_task.cc',
]

torch_ltc_sources = (
    glob.glob('pytorch/lazy_tensor_core/lazy_tensor_core/csrc/*.cpp') +
    glob.glob('pytorch/lazy_tensor_core/lazy_tensor_core/csrc/ops/*.cpp') +
    glob.glob('pytorch/lazy_tensor_core/lazy_tensor_core/csrc/view_ops/*.cpp') +
    glob.glob('pytorch/lazy_tensor_core/lazy_tensor_core/csrc/compiler/*.cpp') +
    glob.glob('pytorch/lazy_tensor_core/lazy_tensor_core/csrc/ts_backend/*.cpp') +
    glob.glob('pytorch/lazy_tensor_core/lazy_tensors/client/*.cc') +
    glob.glob('pytorch/lazy_tensor_core/lazy_tensors/*.cc') +
    glob.glob('pytorch/lazy_tensor_core/lazy_tensors/client/lib/*.cc') +
    glob.glob('pytorch/lazy_tensor_core/lazy_tensors/core/platform/*.cc') +
    client_files)

torch_conversion_sources = [
    os.path.join(base_dir, os.path.pardir, 'pytorch_blade/src/compiler/mlir/converters/mhlo_conversion.cpp')]
torch_disc_sources = glob.glob('torch_disc/csrc/*.cpp') + torch_ltc_sources + torch_conversion_sources
# Constant known variables used throughout this file.
lib_path = os.path.join(base_dir, 'torch_disc/lib')
pytorch_source_path = os.getenv('PYTORCH_SOURCE_PATH', os.path.join(base_dir, 'pytorch'))

include_dirs = [
    base_dir,
    pytorch_source_path,
    os.path.join(pytorch_source_path, 'torch/csrc'),
    os.path.join(pytorch_source_path, 'lazy_tensor_core'),
    'torch_disc/csrc',
    os.path.join(base_dir, os.path.pardir, 'pytorch_blade/src'),
]

extra_compile_args = [
    '-std=c++14',
    '-Wno-sign-compare',
    '-Wno-unknown-pragmas',
    '-Wno-return-type',
]

if re.match(r'clang', os.getenv('CC', '')):
    extra_compile_args += [
        '-Wno-macro-redefined',
        '-Wno-return-std-move',
    ]

library_dirs = []
library_dirs.append(lib_path)
extra_link_args = []


def make_relative_rpath(path):
    return '-Wl,-rpath,$ORIGIN/' + path

def _compile_parallel(self,
                      sources,
                      output_dir=None,
                      macros=None,
                      include_dirs=None,
                      debug=0,
                      extra_preargs=None,
                      extra_postargs=None,
                      depends=None):
    # Those lines are copied from distutils.ccompiler.CCompiler directly.
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    def compile_one(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    list(
        multiprocessing.pool.ThreadPool(multiprocessing.cpu_count()).imap(
            compile_one, objects))
    return objects


def _check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


# Plant the parallel compile function.
if _check_env_flag('COMPILE_PARALLEL', default='1'):
    try:
        if (inspect.signature(distutils.ccompiler.CCompiler.compile) ==
                inspect.signature(_compile_parallel)):
            distutils.ccompiler.CCompiler.compile = _compile_parallel
    except BaseException:
        pass

setup(
    name='torch_disc',
    version='0.1',
    description='DISC backend implementation for Lazy tensors Core',
    url='https://github.com/alibaba/BladeDISC',
    author='DISC Dev Team',
    author_email='disc-dev@alibaba-inc.com',
    # Exclude the build files.
    packages=find_packages(exclude=['build']),
    ext_modules=[
        CppExtension(
            '_DISC',
            torch_disc_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args + \
                [make_relative_rpath('torch_disc/lib')],
        ),
    ],
    package_data={
        'torch_disc': [
            'lib/*.so*',
        ],
    },
    data_files=[
    ])

