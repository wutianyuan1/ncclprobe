from setuptools import setup, find_packages

setup(
    name='control_plane',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'redis>=5.0.4',
        'numpy>=1.26.4',
        'matplotlib>=3.8.0',
        'pandas>=1.5.3',
        'statsmodels>=0.14.2',
        'Rbeast>=0.1.19'
    ],
    entry_points = {
        'console_scripts': [
            'global_controller=control_plane.global_controller:main',
            'local_controller=control_plane.local_controller:start_local_controller',
        ],
    }
)
