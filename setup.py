from setuptools import find_packages, setup

cleanrl_requirements = [
    "torch==1.12.1",
    "torchvision==0.13.1",
    "tensorboard==2.11.2",
    "stable-baselines3",
    "pandas==1.5.3",
    "numpy==1.24.1",
    "matplotlib==3.6.3",
    # "gym==0.21.0",
]

setup(
    name="centralised-marl",
    version="0.0.0",
    description="Simple centralised-marl algorithms",
    author="Ruan de Kock",
    license="Apache License, Version 2.0",
    keywords="multi-agent reinforcement-learning python machine learning",
    packages=find_packages(),
    install_requires=[
        "chex==0.1.5",
        "ma-gym==0.0.12", 
        "jax==0.4.1",
        "jaxlib==0.4.1",
        "dm-haiku==0.0.9",
        "distrax==0.1.2",
    ],
    extras_require={
        "cleanrl": cleanrl_requirements,
    },
    classifiers=[
        "Development Status :: 0",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)