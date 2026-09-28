"""Install the ``safe_humanoid`` Isaac Lab extension as a Python package."""

from __future__ import annotations

import tomllib
from pathlib import Path

from setuptools import find_packages, setup


EXTENSION_ROOT = Path(__file__).resolve().parent
EXTENSION_METADATA = tomllib.loads(
    (EXTENSION_ROOT / "config" / "extension.toml").read_text(encoding="utf-8")
)["package"]


setup(
    name="safe-humanoid",
    version=EXTENSION_METADATA["version"],
    description=EXTENSION_METADATA["description"],
    author=EXTENSION_METADATA["author"],
    maintainer=EXTENSION_METADATA["maintainer"],
    url=EXTENSION_METADATA["repository"],
    keywords=EXTENSION_METADATA["keywords"],
    license="BSD-3-Clause",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11,<3.12",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
)
