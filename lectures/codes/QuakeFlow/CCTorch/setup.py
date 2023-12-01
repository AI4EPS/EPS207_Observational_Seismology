from setuptools import setup

setup(
    name="CCTorch",
    version="0.1.2",
    long_description="Cross-Correlation using Pytorch",
    long_description_content_type="text/markdown",
    packages=["cctorch"],
    install_requires=["torch", "torchvision", "h5py", "matplotlib", "pandas"],
)
