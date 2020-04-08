import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="test-dopeuser",
    version="0.0.1-alpha",
    author="Platon Karpov, Iskandar Sitdikov",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pikarpov-LANL/Sapsan",
    packages=setuptools.find_packages(),
    keywords=['astrophysics'],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    # install_requires=[
    #     'scikit-learn==0.21.3',
    #     'scipy==1.3.1',
    #     'torch==1.3.1',
    #     'torchvision==0.4.2',
    #     'mlflow==1.4.0',
    #     'numpy==1.17.3',
    #     'h5py==2.9.0'
    # ],
    python_requires='>=3.6',
)