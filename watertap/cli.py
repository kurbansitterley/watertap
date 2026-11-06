#################################################################################
# WaterTAP Copyright (c) 2020-2026, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Laboratory of the Rockies, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

import os
import sys
import click
import shutil
import subprocess
import importlib.util
from importlib import metadata


def _try_reaktoro_installation():
    """
    Tries to import and run a Reaktoro-PSE example to check if the installation works.
    """
    from reaktoro_pse.examples import simple_desalination as sd

    sd.main()
    click.echo("Reaktoro-PSE installation test completed.")


def _install_watertap_reaktoro_dependencies():
    """
    Installs the WaterTAP Reaktoro dependencies using pip.
    """
    cmd = [sys.executable, "-m", "pip", "install", "watertap[reaktoro]"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise click.ClickException(
            f"pip installing optional reaktoro dependencies failed with exit code {e.returncode}"
        )


def _version_of(dist_name: str) -> str:
    """
    Returns the version of the specified distribution.
    """
    return metadata.version(dist_name)


def _module_installed(name: str) -> bool:
    """
    Checks if the specified module is installed.
    """
    return importlib.util.find_spec(name) is not None


@click.group()
def watertap_cli():
    pass


@watertap_cli.command("install-reaktoro")
@click.option(
    "--yes", "-y", "yes", is_flag=True, help="Pass -y to conda to auto-confirm."
)
@click.option(
    "--dry-run",
    "-d",
    "dry_run",
    is_flag=True,
    help="Perform a dry run without actually installing packages.",
)
@click.option(
    "--test-install",
    "-t",
    "test_install",
    is_flag=True,
    help="Try importing reaktoro to check if installation works.",
)
def install_reaktoro(yes: bool, dry_run: bool, test_install: bool):
    """
    Installs WaterTAP Reaktoro dependencies
    """

    # First check if optional reaktoro dependencies are installed
    # Right now, this is only reaktoro-pse
    if _module_installed("reaktoro_pse"):
        version = _version_of("reaktoro-pse")
        click.echo(f"reaktoro-pse v{version} already installed.")
    if not _module_installed("reaktoro_pse"):
        if dry_run:
            click.echo("Would install reaktoro-pse.")
        else:
            _install_watertap_reaktoro_dependencies()
            click.echo("Installed reaktoro-pse.")

    # Check if conda-forge dependencies are installed
    needed = ["reaktoro", "cyipopt"]
    available = [_module_installed(m) for m in needed]
    if all(available):
        version_reaktoro = _version_of("reaktoro")
        version_cyipopt = _version_of("cyipopt")
        if test_install:
            # If it is already installed, can test anyway
            _try_reaktoro_installation()
            click.echo("WaterTAP Reaktoro dependencies installation test completed.")
        click.echo(
            f"reaktoro v{version_reaktoro} and cyipopt v{version_cyipopt} already installed; nothing to do."
        )
        return

    conda = shutil.which("conda")
    if not conda:
        raise click.ClickException(
            "conda not found on PATH. Please install/activate Conda and retry."
        )

    # Build conda install command for the active environment
    cmd = [conda, "install", "-c", "conda-forge", "reaktoro", "cyipopt"]
    if yes:
        cmd.append("-y")

    # If running inside an activated conda env, optionally pin the target env
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_prefix:
        cmd.extend(["-p", conda_prefix])
    elif conda_env and conda_env not in ("base", ""):
        cmd.extend(["-n", conda_env])

    if dry_run:
        click.echo(f"Would run conda {' '.join(cmd[1:]) }")
        return

    click.echo(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise click.ClickException(
            f"conda install of reaktoro dependencies failed with exit code {e.returncode}"
        )
    click.echo(f"Ran: {' '.join(cmd)}")

    if test_install:
        _try_reaktoro_installation()
        click.echo("WaterTAP Reaktoro dependencies installation test completed.")

    click.echo("WaterTAP Reaktoro dependencies successfully installed.")


def main():
    watertap_cli()
