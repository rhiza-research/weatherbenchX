# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import setuptools

base_requires = [
]
docs_requires = [
    "myst-nb",
    "myst-parser",
    "sphinx",
    "sphinx_rtd_theme",
]
tests_requires = [
    "pytest",
    "pyink",
    # work around https://github.com/zarr-developers/zarr-python/issues/2963
    "numcodecs<0.16.0",
]

setuptools.setup(
    name="weatherbenchX",
    version="0.0.0",
    license="Apache 2.0",
    author="Google LLC",
    author_email="weatherbenchX@google.com",
    install_requires=base_requires,
    extras_require={
        "tests": tests_requires,
        "docs": docs_requires,
    },
    url="https://github.com/google-research/weatherbenchX",
    packages=setuptools.find_packages(),
)
