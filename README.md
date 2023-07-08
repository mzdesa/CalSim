# CalSim
Multi-purpose educational robotics simulator built for the Berkeley courses 106/206 AB.

## Building Package
To build a new version of the package, first open the pyproject.toml file and increment the version number by 1 - say, for example, that the version number is now 0.0.2. Once you've done this, enter the command "python -m build" in the directory containing pyproject.toml. After this, you're ready to upload your new version to PyPI. To do this, type "twine upload dist/calsim-0.0.2*" to upload version 0.0.2. Note that the star is included such that both the tar.gz and .whl build files associated with V0.0.2 are uploaded.
