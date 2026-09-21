from setuptools import setup, find_packages
import sys
import os
import modulefinder
import shutil

# ------------- Configuration ----------------
PACKAGE_NAME = "pyanal"
VERSION = "0.1"
ENTRY_SCRIPT = os.path.join(PACKAGE_NAME, "__init__.py")
# --------------------------------------------

IGNORED_MODULES = {"__main__", "test", "tests", "setup"}

# Scan your package for imports
finder = modulefinder.ModuleFinder()
finder.run_script(ENTRY_SCRIPT)

# Get standard library modules for current Python version
stdlib = sys.stdlib_module_names

# Filter only external dependencies (ignore built-ins and your own package)
external_deps = sorted(
    name for name in finder.modules
    if not any(name == std or name.startswith(std + ".") for std in stdlib)
    and not name.startswith(PACKAGE_NAME + ".") and not name == PACKAGE_NAME and not name in IGNORED_MODULES
)

# Install_requires list ready to use
install_requires = list(external_deps)

print("Detected install_requires:", install_requires)  # optional debug

# ------------- Setup ----------------
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_packages(),
    install_requires=install_requires,
)

egg_info_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"{PACKAGE_NAME}.egg-info"))
if os.path.exists(egg_info_path):
    shutil.rmtree(egg_info_path)
