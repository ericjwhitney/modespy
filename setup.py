import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modespy",
    version="0.9.0",
    author="Eric J. Whitney",
    author_email="eric.j.whitney1@optusnet.removethispart.com.au",
    description="Parameterised linear multi-step methods for the solution of "
                "Ordinary Differential Equations (ODEs).",
    include_package_data=True,  # <<< Note!
    install_requires=['numpy', 'scipy'],
    keywords='ODE differential equations mathematics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=['numpy', 'scipy'],
    url="https://github.com/ericjwhitney/modespy",
    packages=setuptools.find_packages(include=['modespy', 'modespy.*']),
    python_requires='>=3.7',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Mathematics"
    ]
)
