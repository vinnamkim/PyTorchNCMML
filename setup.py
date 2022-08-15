from setuptools import find_packages, setup

setup(
    name="torchncmml",
    version="0.0.0",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
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
