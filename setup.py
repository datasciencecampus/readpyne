# stdlib
import time
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="readpyne",
    version="0.2.3",
    author="Art",
    author_email="arturas.eidukas@ons.gov.uk",
    description="A package to extract text lines from receipts (and eventually other sources)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={"": ["models/*.pb", "models/*.pkl"]},
    install_requires=install_requires,
    classifiers=["Programming Language :: Python :: 3"],
)
