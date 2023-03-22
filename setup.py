from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='fboal',
    version='0.0.0',
    description='Fixed-budget online adaptive learning for PINNs',
    url='https://github.com/nguyenkhoa0209/FBOAL',
    author='Thi Nguyen Khoa Nguyen',
    author_email='thi.nguyen2@ens-paris-saclay.fr',
    license='BSD-2',
    packages=find_packages(exclude=["tests"]),
    install_requires=["numpy>=1.16", "tensorflow>=2.3", "shapely"],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown'
)