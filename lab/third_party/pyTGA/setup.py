from setuptools import setup, find_packages
import pyTGA

if __name__ == "__main__":
    setup(
        name="pyTGA",

        version=pyTGA.VERSION,

        description='A pure Python module to manage TGA images',
        long_description='A pure Python module to manage TGA images',

        # Author details
        author='Mirco Tracolli',
        author_email='mirco.theone@gmail.com',
        url='https://github.com/MircoT/pyTGA',
        download_url='https://github.com/MircoT/pyTGA/releases/latest',

        # Choose your license
        license='MIT',

        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        classifiers=[
            'Development Status :: 5 - Production/Stable',

            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: End Users/Desktop',

            'License :: OSI Approved :: MIT License',

            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
        ],

        keywords='tga image development',

        packages=find_packages(exclude=['contrib', 'docs', 'tests']),

        install_requires=[],
        extras_require={},
        package_data={},
        data_files=[],
        entry_points={}
    )
