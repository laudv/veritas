#!/usr/bin/env python3

# https://github.com/benjaminjack/python_cpp_example/

import os
import re
import sys
import sysconfig
import platform
import subprocess
import unittest

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from shutil import copyfile, copymode


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


def test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
META = {}
with open(os.path.join("src", "python", "veritas", "__init__.py")) as f:
    locs = {}
    for l in f.readlines():
        m = re.fullmatch('^__(?P<key>\\w+)__\\s*=\\s*(?P<value>.+)\\n$', l)
        if m:
            k, v = m.group("key"), m.group("value")
            exec(f"__{k}__={v}", globals(), locs)
            META[k] = locs[f"__{k}__"]
    del locs

if __name__ == "__main__":
    setup(
        name="veritas",
        license=META["license"],
        python_requires='>=3.6',
        url=META["url"],
        version=META["version"],
        author=META["author"],
        author_email=META["email"],
        maintainer=META["author"],
        maintainer_email=META["email"],
        long_description=META["doc"],
        packages=find_packages('src/python'),
        install_requires=["numpy"],
        package_dir={ "": "src/python" },
        ext_modules=[CMakeExtension("veritas/veritas")],
        cmdclass=dict(build_ext=CMakeBuild),
        test_suite="setup.test_suite",
        zip_safe=False,
    )
