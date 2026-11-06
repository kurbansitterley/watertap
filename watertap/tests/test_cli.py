import pytest
from click.testing import CliRunner
import importlib.util
from importlib import metadata
import subprocess

from watertap.cli import watertap_cli
import sys
import shutil

import pytest
from click.testing import CliRunner

# Import your CLI module (adjust this import to your package structure)
import watertap.cli as wt_cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    # Help shows with no subcommand
    result = runner.invoke(watertap_cli, ["--help"])
    assert result.exit_code == 0
    assert "Commands:" in result.output
    assert "install-reaktoro" in result.output


def test_cli_already_installed(runner, monkeypatch):
    # Make everything look installed
    monkeypatch.setattr(wt_cli, "_module_installed", lambda name: True)

    # Return per-dist versions
    def fake_version_of(name):
        if name == "reaktoro-pse":
            return "0.2.0"
        if name == "reaktoro":
            return "2.5.1"
        if name == "cyipopt":
            return "1.4.0"
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(wt_cli, "_version_of", fake_version_of)
    # Ensure no subprocess calls happen
    called = {"run": 0}

    def fake_run(*args, **kwargs):
        called["run"] += 1
        raise AssertionError("subprocess.run should not be called in this path")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(watertap_cli, ["install-reaktoro"])
    # Because all required modules are already installed, nothing should be done
    assert result.exit_code == 0
    assert "reaktoro-pse v0.2.0 already installed." in result.output
    assert "already installed; nothing to do." in result.output
    assert called["run"] == 0


def test_cli_dry_run_when_not_installed(runner, monkeypatch):
    # Nothing installed
    def fake_module_installed(name):
        return False

    monkeypatch.setattr(wt_cli, "_module_installed", fake_module_installed)

    # Pretend conda is present
    monkeypatch.setattr(
        shutil, "which", lambda name: "/opt/conda/bin/conda" if name == "conda" else None
    )

    # subprocess.run should not be called in dry-run
    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not run in dry-run")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(watertap_cli, ["install-reaktoro", "--dry-run"])
    assert result.exit_code == 0
    assert "Would install reaktoro-pse." in result.output
    assert "Would run conda install -c conda-forge reaktoro cyipopt" in result.output


def test_cli_full_install(runner, monkeypatch):
    # reaktoro_pse and conda-forge packages not installed
    def fake_module_installed(name):
        return False

    monkeypatch.setattr(wt_cli, "_module_installed", fake_module_installed)

    # conda available
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/opt/conda/bin/conda" if name == "conda" else None,
    )

    # Pin to a specific env via CONDA_PREFIX
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/wt")
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

    calls = []

    def fake_run(cmd, check=True):
        # Record the command and pretend success
        calls.append(cmd)

        class R0:
            returncode = 0

        return R0()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(watertap_cli, ["install-reaktoro", "-y"])
    assert result.exit_code == 0
    # First call should be pip install watertap[reaktoro]
    assert calls[0][:3] == [sys.executable, "-m", "pip"]
    assert "watertap[reaktoro]" in calls[0]
    # Second call should be conda install with -c conda-forge reaktoro cyipopt -y -p CONDA_PREFIX
    assert calls[1][0] == "/opt/conda/bin/conda"
    assert calls[1][1:3] == ["install", "-c"]
    assert "conda-forge" in calls[1]
    assert "reaktoro" in calls[1] and "cyipopt" in calls[1]
    assert "-y" in calls[1]
    # Ensure prefix is set
    assert "-p" in calls[1]
    p_index = calls[1].index("-p")
    assert calls[1][p_index + 1] == "/opt/conda/envs/wt"
    # Output contains status lines
    assert "Installed reaktoro-pse." in result.output
    assert "Running: /opt/conda/bin/conda" in result.output


def test_cli_conda_not_found(runner, monkeypatch):
    # reaktoro_pse not installed
    monkeypatch.setattr(wt_cli, "_module_installed", lambda name: False)
    # but no conda on PATH
    monkeypatch.setattr(shutil, "which", lambda name: None)

    # Avoid actually running pip; let it succeed
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)

    result = runner.invoke(watertap_cli, ["install-reaktoro"])
    assert result.exit_code != 0
    assert "conda not found on PATH" in result.output


def test_cli_test_install_flag(runner, monkeypatch):
    # Make reaktoro and cyipopt appear installed so code goes to "already installed" path
    def fake_module_installed(name):
        return name in {"reaktoro_pse", "reaktoro", "cyipopt"}

    monkeypatch.setattr(wt_cli, "_module_installed", fake_module_installed)
    monkeypatch.setattr(wt_cli, "_version_of", lambda name: "1.0.0")

    called = {"probe": 0}

    def fake_probe():
        called["probe"] += 1

    monkeypatch.setattr(wt_cli, "_try_reaktoro_installation", fake_probe)

    result = runner.invoke(watertap_cli, ["install-reaktoro", "--test-install"])
    assert result.exit_code == 0
    assert called["probe"] == 1
    assert "installation test completed" in result.output.lower()
