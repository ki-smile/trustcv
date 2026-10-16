import pytest

from trustcv.cli import EXAMPLES, _path, main


def test_every_registered_example_ships():
    for i in range(1, len(EXAMPLES) + 1):
        assert _path(i).is_file()


def test_examples_list_and_about(capsys):
    assert main(["examples"]) == 0
    assert main(["about"]) == 0
    out = capsys.readouterr().out
    assert "all_cv_methods_demo" in out and "@software" in out


def test_examples_copy_and_bad_number(tmp_path, capsys):
    assert main(["examples", "1", "--copy", str(tmp_path)]) == 0
    assert (tmp_path / f"{EXAMPLES[0][0]}.py").exists()
    assert main(["examples", "99"]) == 2


def test_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
