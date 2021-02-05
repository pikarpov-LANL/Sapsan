import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("version", "r") as f:
    version = f.read()

setuptools.setup(
    name="sapsan",
    version=version,
    author="Platon Karpov, Iskandar Sitdikov",
    author_email="plkarpov@ucsc.edu",
    description="Sapsan project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pikarpov-LANL/Sapsan",
    packages=setuptools.find_packages(),
    keywords=['experiments', 'reproducibility', 'astrophysics'],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'numpy==1.17.3',
        'Click>=7.0',
        'torch==1.5.0',
        'torchvision==0.6.0',
        'catalyst==20.7',
        'h5py==2.9.0',
        'jupyter==1.0.0',
        'matplotlib==3.1.1',
        'mlflow==1.11.0',
        'numpy==1.17.3',
        'pandas==0.25.3',
        'Pillow==6.2.1',
        'plotly==4.3.0',
        'safitty==1.3',
        'scikit-image==0.16.2',
        'scikit-learn==0.21.3',
        'scipy==1.3.1',
        'seaborn==0.9.0',
        'six==1.13.0',
        'hiddenlayer==0.3',
        'graphviz==0.14',
        'streamlit==0.58.0'
    ],
    python_requires='>=3.6',
    entry_points='''
        [console_scripts]
        sapsan=sapsan.core.cli.cli:sapsan
    '''
)


