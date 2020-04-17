import setuptools

with open("README.md", "r") as fh:
    long_description = "TEST README" #fh.read()

setuptools.setup(
    name="test-pypi-dope-release",
    version="0.0.1-alpha",
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    keywords=[''],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'numpy==1.17.3',
        'Click>=6'
    ],
    python_requires='>=3.6',
    entry_points='''
        [console_scripts]
        sapsan=sapsan.core.cli.cli:sapsan
    '''
)


