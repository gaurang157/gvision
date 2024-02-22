from setuptools import setup, find_packages

setup(
    name="gvision",
    version="0.7a3",
    author="Gaurang Ingle",
    author_email="gaurang.ingle@gmail.com",
    description="A package for automated computer vision tasks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={'gvision': ['cli.py', 'pages/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
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
        "streamlit==1.31.0",
        "streamlit_ace==0.1.1",
        "streamlit_webrtc==0.47.1",
        "supervision==0.18.0",
        "torch==2.2.0",
        "ultralytics==8.1.10",
        "YAML2ST==1.0.20"
    ],
    entry_points={
        "console_scripts": [
            "gvision=gvision:main1",
        ],
    },
)
