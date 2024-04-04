import setuptools
import os
import io

current_path = os.path.dirname(os.path.realpath(__file__))

with io.open(f"{current_path}/README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="rulexai",
    version="1.1.0",
    author="Dawid Macha",
    author_email="dawid.m.macha@gmail.com",
    description="RuleXAI is a rule-based aproach to explain the output of any machine learning model. It is suitable for classification, regression and survival tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adaa-polsl/RuleXAI",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "pandas >= 1.5.0, < 2.3.0",
        "numpy ~= 1.26.4",
        "matplotlib ~= 3.8.3",
        "rulekit ~= 1.7.6",
        "lifelines ~= 0.28.0"
    ],
    test_suite="tests",
)
