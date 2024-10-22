import pathlib

from setuptools import find_packages, setup


def get_version():
    """Gets the vmas version."""
    path = CWD / "vni" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


CWD = pathlib.Path(__file__).absolute().parent


setup(
    name="vni",
    version=get_version(),
    description="Vectorized Numerical Interventions",
    url="https://github.com/Giovannibriglia/VectorizedNumericalInterventions",
    license="GPLv3",
    author="Giovanni Briglia",
    author_email="giovanni.briglia@unimore.it",
    packages=find_packages(),
    install_requires=["torch"],
    include_package_data=True,
)