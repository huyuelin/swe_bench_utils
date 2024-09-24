import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import docker
import json
import pytest
import traceback
from unittest.mock import patch, MagicMock
from pathlib import Path

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.docker_utils import (
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    cleanup_container,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    close_logger,
    setup_logger,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec import TestSpec
from swebench.harness.run_evaluation import run_instance, EvaluationError

@pytest.fixture
def mock_docker_client():
    client = MagicMock(spec=docker.DockerClient)
    container = MagicMock()
    container.exec_run.return_value = MagicMock(exit_code=0, output=b"")
    container.start = MagicMock()
    client.containers.run.return_value = container
    return client, container

@pytest.fixture
def mock_test_spec():
    test_spec = MagicMock(spec=TestSpec)
    test_spec.instance_id = "test_instance"
    test_spec.instance_image_key = "test_image_key"
    test_spec.eval_script = "echo 'Running tests'"
    test_spec.env_image_key = "test_env_image_key"
    return test_spec

@pytest.fixture
def mock_prediction():
    return {
        "instance_id": "test_instance",
        "model_name_or_path": "test_model",
        "model_patch": "Test patch content"
    }

@pytest.fixture
def mock_logger():
    return MagicMock()

@patch('swebench.harness.run_evaluation.build_container')
@patch('swebench.harness.run_evaluation.copy_to_container')
@patch('swebench.harness.run_evaluation.exec_run_with_timeout')
@patch('swebench.harness.run_evaluation.get_eval_report')
@patch('swebench.harness.run_evaluation.cleanup_container')
@patch('swebench.harness.run_evaluation.remove_image')
@patch('swebench.harness.run_evaluation.close_logger')
@patch('builtins.open', new_callable=MagicMock)
def test_run_instance(mock_open, mock_close_logger, mock_remove_image, mock_cleanup_container, 
                      mock_get_eval_report, mock_exec_run_with_timeout, 
                      mock_copy_to_container, mock_build_container,
                      mock_docker_client, mock_test_spec, mock_prediction, mock_logger, tmpdir):
    client, container = mock_docker_client
    mock_build_container.return_value = container
    mock_exec_run_with_timeout.return_value = ("Test output", False, 10.0)
    mock_get_eval_report.return_value = {
        "test_instance": {"resolved": True}
    }

    run_id = "test_run"
    log_dir = Path(tmpdir) / RUN_EVALUATION_LOG_DIR / run_id / "test_model" / "test_instance"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Mock writing to report.json
    mock_open.return_value.__enter__.return_value.write.return_value = None

    result = run_instance(
        test_spec=mock_test_spec,
        pred=mock_prediction,
        rm_image=False,
        force_rebuild=False,
        client=client,
        run_id=run_id,
        timeout=1800
    )

    # Assertions
    assert result[0] == "test_instance"
    assert result[1]["test_instance"]["resolved"] == True

    mock_build_container.assert_called_once()
    container.start.assert_called_once()
    assert mock_copy_to_container.call_count == 2  # For patch and eval script
    container.exec_run.assert_called()
    mock_exec_run_with_timeout.assert_called_once()
    mock_get_eval_report.assert_called_once()
    mock_cleanup_container.assert_called_once()
    mock_close_logger.assert_called_once()

    # Check if report.json was "created" (mocked)
    mock_open.assert_called_with(log_dir / "report.json", "w")
    mock_open.return_value.__enter__.return_value.write.assert_called_once()

@patch('swebench.harness.run_evaluation.build_container')
@patch('swebench.harness.run_evaluation.setup_logger')
def test_run_instance_build_error(mock_setup_logger, mock_build_container, 
                                  mock_docker_client, mock_test_spec, mock_prediction, tmpdir):
    client, _ = mock_docker_client
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    mock_build_container.side_effect = BuildImageError("Test error", "test_instance", mock_logger)

    run_id = "test_run"
    log_dir = Path(tmpdir) / RUN_EVALUATION_LOG_DIR / run_id / "test_model" / "test_instance"
    log_dir.mkdir(parents=True, exist_ok=True)

    result = run_instance(
        test_spec=mock_test_spec,
        pred=mock_prediction,
        rm_image=False,
        force_rebuild=False,
        client=client,
        run_id=run_id,
        timeout=1800
    )

    assert result is None
    mock_logger.info.assert_called_with(traceback.format_exc())

@patch('swebench.harness.run_evaluation.build_container')
@patch('swebench.harness.run_evaluation.copy_to_container')
@patch('swebench.harness.run_evaluation.setup_logger')
def test_run_instance_patch_fail(mock_setup_logger, mock_copy_to_container, mock_build_container, 
                                 mock_docker_client, mock_test_spec, mock_prediction, tmpdir):
    client, container = mock_docker_client
    mock_logger = MagicMock()
    mock_setup_logger.return_value = mock_logger
    mock_build_container.return_value = container
    container.exec_run.return_value = MagicMock(exit_code=1, output=b"Patch failed")

    run_id = "test_run"
    log_dir = Path(tmpdir) / RUN_EVALUATION_LOG_DIR / run_id / "test_model" / "test_instance"
    log_dir.mkdir(parents=True, exist_ok=True)

    result = run_instance(
        test_spec=mock_test_spec,
        pred=mock_prediction,
        rm_image=False,
        force_rebuild=False,
        client=client,
        run_id=run_id,
        timeout=1800
    )

    assert result is None
    mock_logger.info.assert_any_call(f"{APPLY_PATCH_FAIL}:\nPatch failed")

if __name__ == "__main__":
    pytest.main()
    