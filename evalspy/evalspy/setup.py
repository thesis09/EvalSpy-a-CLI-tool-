from setuptools import setup, find_packages

setup(
    name="evalspy",
    version="0.1.0",
    author="Kaustubh Kubitkar",
    author_email="kaustubhkubitkar@gmail.com",
    description="Audit your LLM benchmark evaluation pipeline for known failure modes before you waste compute.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thesis09/evalspy",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],   # zero dependencies — pure stdlib
    entry_points={
        "console_scripts": [
            "evalspy=evalspy.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
