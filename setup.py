from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="perf_lgdt",
    version="1.0",
    rust_extensions=[RustExtension("perf_lgdt", binding=Binding.PyO3)],
    # packages=["lgdt_python"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
