# This file is part of ts_weathernbeats.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""nox configuration for the weathernbeats service."""

import nox
from nox_uv import session

# Default sessions.
nox.options.sessions = ["lint", "typing", "test"]

# Other nox defaults.
nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

# Settings required by the configuration model. The tests replace bundle
# loading and the EFD client, so the bundle path is never read.
TEST_ENV = {
    "WEATHERNBEATS_BUNDLE": "tests/fake-bundle",
    "WEATHERNBEATS_SIMULATION": "true",
}


@session(name="coverage-report", uv_groups=["dev"])
def coverage_report(session: nox.Session) -> None:
    """Generate a code coverage report from the test suite."""
    session.run("coverage", "report", *session.posargs)


@session(uv_only_groups=["lint"], uv_no_install_project=True)
def lint(session: nox.Session) -> None:
    """Run Ruff and check that uv.lock is up to date."""
    session.run("ruff", "check", *session.posargs)
    session.run("ruff", "format", "--check")
    session.run("uv", "lock", "--check", external=True)


@session
def run(session: nox.Session) -> None:
    """Run a local development server."""
    session.run(
        "uvicorn",
        "weathernbeats_service.main:app",
        "--reload",
    )


@session(uv_groups=["dev"])
def test(session: nox.Session) -> None:
    """Test the weathernbeats service."""
    session.run(
        "pytest",
        "--cov=weathernbeats_service",
        "--cov-branch",
        "--cov-report=",
        *session.posargs,
        env=TEST_ENV,
    )


@session(uv_groups=["dev", "typing"])
def typing(session: nox.Session) -> None:
    """Run mypy."""
    session.run(
        "mypy",
        *session.posargs,
        "noxfile.py",
        "src",
        "tests",
    )
