from setuptools import find_packages, setup
import os
from glob import glob

package_name = "crop_row_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"),
            glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Amey Gavale",
    maintainer_email="agavale2@illinois.edu",
    description="Multi-branch crop row perception for Amiga navigation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fusion_node = crop_row_perception.fusion_node:main",
        ],
    },
)
