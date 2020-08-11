from shutil import copyfile

import subprocess
from distutils.command.install import install as _install  # type: ignore
import setuptools


# This class handles the pip install mechanism.
class install(_install):  # pylint: disable=invalid-name
    sub_commands = _install.sub_commands + [("CustomCommands", None)]


class CustomCommands(setuptools.Command):
    """A setuptools Command class able to run arbitrary commands."""

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def install_tf2_object_detection(self):
        """
        This will install object detection API into the production cluster
        if you are doing development, you will prefer to install it using
        pip install -e . so it is in development mode
        (you might need to change parts of the API)
        """
        try:
            import object_detection

            print("\n\nObject detection API found\n\n")
        except ImportError:
            # Clone the modified tensorflow models repository if it doesn't already exist
            command = "git clone --depth 1 https://github.com/jgaz/models /tmp/models".split(
                " "
            )
            subprocess.run(command, check=False)

            command = "protoc object_detection/protos/*.proto --python_out=.".split(" ")
            subprocess.run(command, check=False, cwd="/tmp/models/research/")

            copyfile(
                "/tmp/models/research/object_detection/packages/tf2/setup.py",
                "/tmp/models/research/setup.py",
            )
            command = "python -m pip install .".split(" ")
            subprocess.run(command, check=True, cwd="/tmp/models/research/")

    def run(self):
        self.install_tf2_object_detection()


setuptools.setup(
    package_dir={"": "src"},
    cmdclass={"install": install, "CustomCommands": CustomCommands},
)
