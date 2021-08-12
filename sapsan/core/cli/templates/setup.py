TEMPLATE = """
import setuptools
from {name}._version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as f:
    requirements = [i.strip() for i in f.readlines()]

setuptools.setup(
    name="{name}",
    version=__version__,
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    keywords=[],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=requirements,
    python_requires='>=3.7, <3.9',
)
"""

def get_setup_template(name: str) -> str:
    return TEMPLATE.format(name=name)