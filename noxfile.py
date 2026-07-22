"""Nox configuration file for running tests against multiple framework versions.

This file defines test runs to test the veritas converters against different
versions of xgboost, lightgbm, and scikit-learn across supported Python
interpreters.

Usage:
  - Run all compatible test configurations:
        nox

  - Run only xgboost tests for all versions:
        nox -s tests_xgboost

  - Run a specific version combination (forcing a specific Python version if needed):
        nox -s "tests_xgboost(xgboost='3.2.0')" --force-python 3.14
"""

import nox

# Force nox to skip missing Python interpreters instead of failing
nox.options.error_on_missing_interpreters = False

# Python versions supported by the project (pyproject.toml: requires-python >= 3.11)
PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]

# Compatibility mapping: defines which xgboost/ligtgbm/sklearn versions to test per Python version.
XGBOOST_BY_PYTHON = {
    "3.11": ["3.2.0"],
    "3.12": ["3.3.0"],
    "3.13": ["3.3.0"],
    "3.14": ["3.3.0"],
}
LIGHTGBM_BY_PYTHON = {
    "3.11": ["4.6.0"],
    "3.12": ["4.6.0"],
    "3.13": ["4.6.0"],
    "3.14": ["4.6.0"],
}
SKLEARN_BY_PYTHON = {
    "3.11": ["1.9.0"],
    "3.12": ["1.9.0"],
    "3.13": ["1.9.0"],
    "3.14": ["1.9.0"],
}


def get_unique_versions(by_python_dict):
    """Extract list of unique version strings from a python-compatibility dictionary."""
    return list(dict.fromkeys(v for versions in by_python_dict.values() for v in versions))


# Dynamically derive global lists of versions to parameterize nox sessions
XGBOOST_VERSIONS = get_unique_versions(XGBOOST_BY_PYTHON)
LIGHTGBM_VERSIONS = get_unique_versions(LIGHTGBM_BY_PYTHON)
SKLEARN_VERSIONS = get_unique_versions(SKLEARN_BY_PYTHON)


def install_veritas(session):
    """Install veritas build and test dependencies."""
    session.install("scikit-build-core")
    session.run("pip", "install", ".[test]")


def run_segmented_tests(
    session,
    package_name,
    version,
    compatibility_dict,
    all_versions,
    marker,
):
    """Helper method to isolate, provision, and run segmented test sessions.

    1. Skips the run if the combination is not in the compatibility matrix.
    2. Builds/installs the current project environment.
    3. Installs the specific package version to override any default.
    4. Invokes pytest targeting the designated framework tests.
    """
    if version not in compatibility_dict.get(session.python, all_versions):
        session.skip(f"{package_name} {version} is not supported/tested on Python {session.python}")

    install_veritas(session)
    session.install(f"{package_name}=={version}")
    session.run("pytest", "-m", marker, "tests/")


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("xgboost", XGBOOST_VERSIONS)
def tests_xgboost(session, xgboost):
    """Test compatibility against different versions of xgboost."""
    run_segmented_tests(
        session,
        "xgboost",
        xgboost,
        XGBOOST_BY_PYTHON,
        XGBOOST_VERSIONS,
        "xgboost",
    )


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("lightgbm", LIGHTGBM_VERSIONS)
def tests_lightgbm(session, lightgbm):
    """Test compatibility against different versions of lightgbm."""
    run_segmented_tests(
        session,
        "lightgbm",
        lightgbm,
        LIGHTGBM_BY_PYTHON,
        LIGHTGBM_VERSIONS,
        "lightgbm",
    )


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("sklearn", SKLEARN_VERSIONS)
def tests_sklearn(session, sklearn):
    """Test compatibility against different versions of scikit-learn."""
    run_segmented_tests(
        session,
        "scikit-learn",
        sklearn,
        SKLEARN_BY_PYTHON,
        SKLEARN_VERSIONS,
        "sklearn",
    )


@nox.session(python=PYTHON_VERSIONS)
def tests_general(session):
    """Run all core/verifier tests not bound to framework-specific integration converters."""
    install_veritas(session)
    session.run("pytest", "-m", "not (xgboost or lightgbm or sklearn)", "tests/")


@nox.session
def lint(session):
    """Lint python source code using ruff."""
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")
