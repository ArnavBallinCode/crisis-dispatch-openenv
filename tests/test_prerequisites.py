from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def test_required_files_exist(project_root: Path) -> None:
    assert (project_root / "README.md").is_file()
    assert (project_root / "requirements.txt").is_file()
    assert (project_root / "openenv.yaml").is_file()
    assert (project_root / "app").is_dir()

    dockerfile_exists = (project_root / "Dockerfile").is_file() or (
        project_root / "server" / "Dockerfile"
    ).is_file()
    assert dockerfile_exists


@pytest.mark.docker
def test_docker_build_if_enabled(project_root: Path) -> None:
    if not _enabled("RUN_DOCKER_BUILD"):
        pytest.skip("Set RUN_DOCKER_BUILD=1 to enable docker build test")

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker command is not available")

    context = project_root if (project_root / "Dockerfile").is_file() else project_root / "server"

    completed = subprocess.run(
        [docker, "build", str(context)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.integration
def test_openenv_validate_if_enabled(project_root: Path) -> None:
    if not _enabled("RUN_OPENENV_VALIDATE"):
        pytest.skip("Set RUN_OPENENV_VALIDATE=1 to enable openenv validate test")

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        pytest.skip("HF_TOKEN not set")

    openenv_cmd = shutil.which("openenv")
    if openenv_cmd:
        cmd = [openenv_cmd, "validate"]
    elif shutil.which("uvx"):
        cmd = ["uvx", "--from", "openenv-core>=0.2.0", "openenv", "validate"]
    else:
        pytest.skip("Neither openenv nor uvx is available")

    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token

    completed = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
