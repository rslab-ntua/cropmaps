import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name = "cropmaps",
    version = "0.0.1.beta",
    author = "Alekos Falagas",
    author_email = "alek.falagas@gmail.com",
    description = "Crop type mapping toolbox.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url="mago.gg",
    packages = setuptools.find_packages(),
    license="GNU General Public License v3 or later (GPLv3+)",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
    python_requires = '>=3.6',
    install_requires=['wheel',
                      'creodias_finder @ git+https://github.com/DHI-GRAS/creodias-finder@main',
                      required],
)