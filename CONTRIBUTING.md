# Contributing

## Increment version

Following [semantic versioning](https://semver.org/) conventions, update the version in `pyproject.toml`

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
>    * MAJOR version when you make incompatible API changes,
>    * MINOR version when you add functionality in a backwards compatible manner, and
>    * PATCH version when you make backwards compatible bug fixes.

## Update `CHANGELOG.md`

Add an additional entry in the `CHANGELOG.md` describing the changes made 
in the new version. When possible, make reference to specific commits or pull requests and 
tag changes with labels like `[bugfix]` or `[feature]`

## Write Docs

Document any new functionality. Since we don't have formal docs, either write this in the `README.md` or add a new
markdown file to `docs/` and link from README.

## Update Dependencies

if any additional libraries were added or removed, add them to the `pyproject.toml` file, either with poetry
(`poetry add <package name>`) or by [specifying it manually](https://python-poetry.org/docs/dependency-specification/).

Then update the lockfile: `poetry lock`

## Write Tests

Currently, the tests are relatively informal, consisting of just running the `dlc-live-test` entrypoint 
(`dlclive.check_install.check_install.main`). Add an additional function (ideally, rather than making one
extremely long `main` function) that tests any new features.