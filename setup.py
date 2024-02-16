from traceback import extract_tb

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Labeeb",
    version="0.0.7",
    author="Mohamed Eldesouki",
    author_email="labeeb@eldesouki.com",
    description="A machine learning engine designed and developed to be both easy to use and source code readable.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/disooqi/Labeeb",
    project_urls={
        "Bug Tracker": "https://github.com/disooqi/Labeeb/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Academic License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=['numpy'],
    extras_require=dict(tests=['pytest'])

)
