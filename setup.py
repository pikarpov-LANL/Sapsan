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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
        'numpy>=1.19.2',
        'Click>=7.1.2',
        'torch>=1.7.1',
        'torchvision>=0.8.2',
        'catalyst>=20.7',
        'h5py>=2.10.0',
        'jupyter>=1.0.0',
        'matplotlib==3.3.2',
        'mlflow==1.11.0',
        'pandas>=1.2.0',
        'Pillow>=8.1.0',
        'plotly==4.3.0',
        'safitty>=1.3',
        'scikit-image==0.17.2',
        'scikit-learn==0.23.2',
        'scipy>=1.5.2',
        'seaborn==0.11.1',
        'six==1.15.0',
        'hiddenlayer==0.3',
        'graphviz==0.14',
        'streamlit==0.58.0'
    ],
    python_requires='>=3.7',
    entry_points='''
        [console_scripts]
        sapsan=sapsan.core.cli.cli:sapsan
    '''
)


