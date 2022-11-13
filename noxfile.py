import nox


nox.options.sessions = [
    "tests",
    "lint",
    "safety",
    "mypy",
    "pytype",
    "typeguard",
    "xdoctest",
    "docs",
]
# python_versions = [
#     "3.10",
# ]
python_versions = ["3.10", "3.9", "3.8", "3.7"]


@nox.session(python=python_versions)
def tests(session):
    session.install(".")
    session.install(
        "pytest", "numpy", "taichi", "taichi-glsl", "coverage[toml]", "pytest-cov", "click"
    )
    session.run("pytest", "--cov")


@nox.session(python=python_versions)
def safety(session):
    """Scan dependencies for insecure packages."""
    requirements = "./requirements.txt"
    session.install("safety")
    session.run("safety", "check", "--full-report", f"--file={requirements}")


locations = "src", "tests", "noxfile.py", "docs/conf.py"


@nox.session(python=python_versions)
def lint(session):
    args = session.posargs or locations
    session.install(
        "flake8",
        # "flake8-annotations",
        # "flake8-bandit",
        # "flake8-black",
        # "flake8-bugbear",
        # "flake8-import-order",
        # "flake8-docstrings",
        # "flake8-rst-docstrings",
        "darglint",
    )
    session.run("flake8", *args)


@nox.session(python=python_versions)
def mypy(session):
    args = session.posargs or locations
    session.install("mypy")
    session.run("mypy", *args)


@nox.session(python="3.7")
def pytype(session):
    """Run the static type checker."""
    args = session.posargs or locations
    session.install("mypy")
    session.run("mypy", *args)


package = "TaichiRender"


@nox.session(python=["3.10"])
def typeguard(session):
    args = session.posargs or ["-m", "not e2e"]
    session.install(".")
    session.install("-r", "./requirements.txt")
    session.install("pytest", "pytest-mock", "typeguard", "click")
    session.run("pytest", f"--typeguard-packages={package}", *args)


@nox.session(python=["3.10"])
def xdoctest(session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.install(".")
    session.install("-r", "./requirements.txt")
    session.install("xdoctest")
    session.run("python", "-m", "xdoctest", package, *args)


@nox.session(python="3.10")
def docs(session) -> None:
    """Build the documentation."""
    session.install("sphinx", "sphinx-autodoc-typehints")
    session.run("sphinx-build", "docs", "docs/_build")
