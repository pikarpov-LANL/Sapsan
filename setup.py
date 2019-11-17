from distutils.core import setup

setup(
    name='sapsan',
    packages=['sapsan'],
    version='0.1',
    license='BSD 3-clause',
    description='',
    author='Platon Karpov, Iskandar Sitdikov',
    author_email='',
    url='https://github.com/pikarpov-LANL/Sapsan',
    download_url='https://github.com/pikarpov-LANL/Sapsan/archive/v_01.tar.gz',
    keywords=['astrophysics'],
    install_requires=[
        'scikit-learn==0.21.3',
        'scipy==1.3.1',
        'torch==1.3.1',
        'torchvision==0.4.2',
        'mlflow==1.4.0',
        'numpy==1.17.3',
        'h5py==2.9.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Physicists',
        'Topic :: Software Development :: Astrophysics',
        'License :: OSI Approved :: BSD 3-clause',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
