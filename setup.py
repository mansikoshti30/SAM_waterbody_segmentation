from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sam-water-segmentation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Water body segmentation from satellite imagery using SAM (Segment Anything Model)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SAM_project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "sam-water-segment=test_sam_5_overall:main_batch",
        ],
    },
    keywords="satellite-imagery water-segmentation SAM deep-learning remote-sensing",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/SAM_project/issues",
        "Source": "https://github.com/yourusername/SAM_project",
        "Documentation": "https://github.com/yourusername/SAM_project#readme",
    },
)
