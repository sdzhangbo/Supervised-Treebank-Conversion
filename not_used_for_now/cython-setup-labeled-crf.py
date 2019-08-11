
import setuptools
import sys

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    sys.exit("""Could not import Cython, which is required to build benepar extension modules.
Please install cython and numpy prior to installing benepar.""")

try:
    import numpy as np
except ImportError:
    sys.exit("""Could not import numpy, which is required to build the extension modules.
Please install cython and numpy prior to installing benepar.""")


setuptools.setup(
    name="benepar",
    version="0.0.1",
    author="Nikita Kitaev",
    author_email="kitaev@cs.berkeley.edu",
    description="Berkeley Neural Parser",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/nikitakit/self-attentive-parser",
    packages=setuptools.find_packages(),
    package_data={'': ['*.pyx']},
    ext_modules = cythonize("labeled_crf_loss.pyx"),
    include_dirs=[np.get_include()],
    classifiers=(
        'Programming Language :: Python :: 2.7',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ),
    setup_requires = ["cython", "numpy"],
)
