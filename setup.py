import os
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as f:
    long_description = f.read()

# read the contents of requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name = 'dcxfm',
    version = '0.0.1',
    author = 'Fuying Wang',
    author_email = 'fuyingw@connect.hku.hk',
    description = 'Dense Chest X-ray Foundation Model',
    url = 'https://github.com/fuying-wang/CXRSeg',
    keywords=['X-ray','foundation model','zero-shot-segmentation'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['test']),
    # install_requires=requirements
)