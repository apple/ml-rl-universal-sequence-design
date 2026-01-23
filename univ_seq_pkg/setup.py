#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import shutil
import os

class CustomBuildExt(build_ext):
    def run(self):
        build_ext.run(self) # Call the original build_ext command

        for extension in ['polar']:
            # move binary to target
            target_directory = 'univ_seq/cpp'
            file_name = self.get_ext_filename(extension)
            built_extension_path = os.path.join(self.build_lib, file_name)
            target_path = os.path.join(target_directory, file_name)
            shutil.move(built_extension_path, target_path)
            # remove extension at root dir
            if os.path.exists(file_name):
                os.remove(file_name)

ext_modules = [
        Pybind11Extension(
             "polar",
            ["univ_seq/cpp/polar.cpp",
             "univ_seq/cpp/PolarCode.cpp"],
            ),
    ]

setup(
    name='univ_seq',
    version='0.1.2',
    packages=find_packages(),
    package_dir={
        "": ".",
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
)
