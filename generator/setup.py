from setuptools import setup

setup(
    use_scm_version={"write_to": "src/version.py"}, package_dir={"": "src"},
)
