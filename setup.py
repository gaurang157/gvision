from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gvision",
    version="0.9",
    author="Gaurang Ingle",
    author_email="gaurang.ingle@gmail.com",
    description="End-to-end automation platform for computer vision projects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gaurang157/gvision",
    packages=find_packages(),
    include_package_data=True,
    package_data={'gvision': ['cli.py', 'pages/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: User Interfaces",
        "Topic :: Utilities"
    ],
    python_requires='>=3.8',
    install_requires=[
        "av==11.0.0",
        "numpy==1.24.4",
        "opencv_python==4.9.0.80",
        "opencv_python_headless==4.8.0.74",
        "pandas==2.0.3",
        "Pillow==10.2.0",
        "PyYAML==6.0.1",
        "Requests==2.31.0",
        "roboflow==1.1.1",
        "streamlit==1.31.1",
        "streamlit_ace==0.1.1",
        "streamlit_webrtc==0.47.1",
        "supervision==0.18.0",
        "torch==2.2.0",
        "ultralytics==8.1.10",
        "YAML2ST==1.0.20",
        "tensorboard>=2.10.0"
    ],
    entry_points={
        "console_scripts": [
            "gvision=gvision:main1",
        ],
    },
    license="MIT",
    keywords=[
        "computer vision", "automation", "model training", "model deployment",
        "object detection", "segmentation", "classification", "pose estimation",
        "deep learning", "machine learning", "Roboflow",
        "Ultralytics", "TensorFlow", "TensorBoard", "Streamlit", "CLI interface", "UI interface"
    ],
    maintainer="Gaurang Ingle",
    maintainer_email="gaurang.ingle@gmail.com",
    project_urls={
        "Bug Reports": "https://github.com/gaurang157/gvision/issues",
        "Source": "https://github.com/gaurang157/gvision",
        "Documentation": "https://github.com/gaurang157/gvision/blob/main/README.md",
        "Say Thanks!": "https://github.com/gaurang157/gvision/issues/new?assignees=&labels=&template=thanks.yml",
    },
)
