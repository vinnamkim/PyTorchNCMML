from setuptools import find_packages, setup

setup(
    name="torchncmml",
    version="0.0.0",
    author="Vinnam Kim",
    author_email="vinnamkim@gmail.com",
    description="A PyTorch implementation of Nearest Class Mean Metric Learning (NCMML)",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="pytorch metric-learning",
    license="MIT",
    url="https://github.com/vinnamkim/PyTorchNCMML",
    packages=find_packages(
        exclude=['example']
    ),
    install_requires=[
        "torch >= 1.5.0",
        "sklearn",
        "matplotlib"
    ],
    python_requires=">=3.6.0",
)
