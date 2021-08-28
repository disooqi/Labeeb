from traceback import extract_tb

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Natasy",
    version="0.0.1",
    author="Mohamed",
    author_email="mohamed@eldesouki.com",
    description="A machine learning engine designed and developed to be both easy to use and source code readable.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/disooqi/Natasy",
    project_urls={
        "Bug Tracker": "https://github.com/disooqi/Natasy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Academic License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    extras_require=dict(tests=['pytest'])

)