import io
import os

from setuptools import setup, find_packages

packages = [p for p in find_packages() if "tests" not in p and "debug" not in p]

root = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(root, "projection-net", "__about__.py"), encoding="utf8") as f:
    about = dict()
    exec(f.read(), about)

with io.open(os.path.join(root, "README.rst"), encoding="utf8") as f:
    readme = f.read()

required = [
    "enum34>=1.1,<2.0; python_version<'3.4'",
    "future>=0.16,<0.17",
    "numpy>=1.15,<1.16",
    "scipy>=1.0,<2.0",
    "scikit-learn>=0.19,<0.20",
    "sklearn-crfsuite>=0.3.6,<0.4",
    "semantic_version>=2.6,<3.0",
    "num2words>=0.5.6,<0.6",
    "plac>=0.9.6,<1.0",
    "requests>=2.0,<3.0",
    "pathlib==1.0.1; python_version < '3.4'",
    "pyaml>=17,<18",
    "deprecation>=2,<3",
    "funcsigs>=1.0,<2.0; python_version < '3.4'"
]

extras_require = {
    "doc": [
        "sphinx>=1.8,<1.9",
        "sphinxcontrib-napoleon>=0.6.1,<0.7",
        "sphinx-rtd-theme>=0.2.4,<0.3",
        "sphinx-tabs>=1.1,<1.2"
    ],
    "test": [
        "mock>=2.0,<3.0",
        "pylint<2",
        "coverage>=4.4.2,<5.0"
    ]
}

setup(name=about["__title__"],
      description=about["__summary__"],
      long_description=readme,
      version=about["__version__"],
      author=about["__author__"],
      author_email=about["__email__"],
      license=about["__license__"],
      url=about["__uri__"],
      install_requires=required,
      extras_require=extras_require,
      classifiers=[
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
      ],
      packages=packages,
      include_package_data=True,
      entry_points={
          "console_scripts": [
              "projection-net=projection_net.__main__:main"
          ]
      },
      zip_safe=False)
