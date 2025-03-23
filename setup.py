from setuptools import setup, find_packages

setup(
    name="LicensePlateRecognition",
    version="0.1.0",
    description="A daemon for license plate recognition from a live camera feed",
    author="Mateusz Kadula",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "licenseplated=licenseplate.main:main",
        ],
    },
    install_requires=[
        "pyyaml",
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "ultralytics",
        "pydantic",
        "opencv-python",
        "easyocr",
        # "picamera2",
    ],
)
