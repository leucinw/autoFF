import os
import shutil
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, 'examples')
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')

sys.path.insert(0, REPO)


@pytest.fixture
def examples_dir():
    return EXAMPLES


@pytest.fixture
def fixtures_dir():
    return FIXTURES


@pytest.fixture
def example_run(tmp_path):
    """Copy an example's inputs and config into an isolated working directory."""
    def _make(name):
        dest = tmp_path / name
        dest.mkdir(parents=True)
        src = os.path.join(EXAMPLES, name)
        shutil.copytree(os.path.join(src, 'input'), dest / 'input')
        shutil.copy(os.path.join(src, 'config.yaml'), dest / 'config.yaml')
        return dest
    return _make
