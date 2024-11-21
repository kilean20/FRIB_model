from setuptools import setup, find_packages

setup(name='FRIB_model',
    description='tools for modeling FRIB',
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Programming Language :: Python :: 3.6',
    'Topic :: FRIB modeling'
    ],
    keywords = ['model calibration', 'machine-learning', 'optimization'],
    author='Kilean Hwang',
    author_email='hwang@frib.msu.edu',
#     license='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'phantasy',
#        'flame_utils',
    ],
    zip_safe=False)