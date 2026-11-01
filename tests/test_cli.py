import json
import pytest
from qiankunpt.cli import main


def test_example_cli_is_offline_and_refuses_to_replace_inputs(tmp_path, capsys):
    output = tmp_path/"demo"
    assert main(["make-example", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["exact_E2_external"] == -.5
    with pytest.raises(FileExistsError):
        main(["make-example", str(output)])


def test_cli_does_not_overwrite_existing_result(tmp_path):
    output = tmp_path/"result.json"
    output.write_text('{"keep":true}')
    with pytest.raises(FileExistsError):
        main(["run", "--fcidump", "not-present", "--reference", "not-present", "--output", str(output)])
    assert json.loads(output.read_text()) == {"keep": True}
