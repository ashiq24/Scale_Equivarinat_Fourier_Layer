from setuptools import setup, find_packages

setup(
    name='scale_equivariant_fourier_layer',  # changed to use underscores
    version='0.1.0',
    author='Md Ashiqur Rahman',
    author_email='rahman79@purdue.edu',
    description='A scale-equivariant Fourier layer for deep learning with PDEs and images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/scale-equivariant-fourier-layer',
    packages=find_packages(include=['scale_eq', 'scale_eq.*']),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.7',
)
