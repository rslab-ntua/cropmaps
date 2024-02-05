import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name = "cropmaps",
    version = "0.0.1.a0",
    author = "Alekos Falagas",
    author_email = "alek.falagas@gmail.com",
    description = "Crop type mapping toolbox.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url="https://cropmaps.readthedocs.io/en/latest/",
    packages = setuptools.find_packages(),
    license="GNU General Public License v3 or later (GPLv3+)",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
    python_requires = '>=3.9',
    install_requires=['wheel',
                      'creodias_finder @ git+https://github.com/DHI-GRAS/creodias-finder@main',
                      required],
)