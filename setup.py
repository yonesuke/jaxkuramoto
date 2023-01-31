import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jaxkuramoto",
    version="0.0.7",
    install_requires=[
        "jax",
    ],
    author="Ryosuke Yoneda",
    author_email="13e.e.c.13@gmail.com",
    description="JAX implementation of Kuramoto model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yonesuke/jaxkuramoto",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)