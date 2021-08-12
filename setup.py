import setuptools
from sapsan._version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as f:
    requirements = [i.strip() for i in f.readlines()]
    
setuptools.setup(
    name="sapsan",
    version=__version__,
    author="Platon Karpov, Iskandar Sitdikov",
    author_email="plkarpov@ucsc.edu",
    description="Sapsan project",
	license = "BSD",
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
    install_requires=requirements,
    python_requires='>=3.7, <3.9',
    include_package_data = True,
    entry_points='''
        [console_scripts]
        sapsan=sapsan.core.cli.cli:sapsan
    '''
)
