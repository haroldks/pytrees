from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="lgdt_optimizer",
    version="1.0",
    rust_extensions=[RustExtension("lgdt_optimizer", binding=Binding.PyO3)],
    packages=find_packages(),
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
