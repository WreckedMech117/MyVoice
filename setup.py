"""Setup script for MyVoice application."""

import os
from setuptools import setup, find_packages

# Read long description from README if available
long_description = ""
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="myvoice",
    version="0.1.0",
    description="Voice cloning desktop application using Qwen3-TTS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MyVoice Development Team",
    url="https://github.com/yourusername/myvoice",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "PyQt6>=6.6.0",
        "requests>=2.31.0",
        "openai-whisper>=20231117",
        "torch>=2.0.0",
        "pyaudio>=0.2.13",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-qt>=4.2.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "myvoice=myvoice.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="voice-cloning tts text-to-speech qwen3-tts audio",
)