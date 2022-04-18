from setuptools import setup, find_packages

install_requires = [
    'scipy',
    'numpy',
    'torch',
    'opencv-python',
    'tqdm',
    'taichi==0.7.26',
    'gym',
    'tensorboard',
    'yacs',
    'baselines',
    'pandas',
    'seaborn',
    'imageio',
    'open3d',
    'lxml',
    'networkx',
    'pytorch3d',
    'pymcubes',
    'python-fcl',
    'pytest'
]

setup(
    name='plb',
    version='0.0.1',
    install_requires=install_requires,
    packages=find_packages(exclude='protocol')
)
