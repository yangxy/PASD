from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as f:
        return f.readlines()


setup(
    name="pasd",
    version="0.0.1",
    url="https://github.com/yangxy/PASD.git",
    description=(
        "[ECCV2024] Pixel-Aware Stable Diffusion for Realistic "
        "Image Super-Resolution and Personalized Stylization"
    ),
    packages=find_packages(),
    install_requires=read_requirements(),
)