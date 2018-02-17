import os
import pytest
from utils.utils import tee

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))


class TestUtils(object):
    def test_tee(self):
        text = 'hello_world'
        output_file = os.path.join(ROOT_DIR, 'output.tee')
        with open(output_file, 'w') as f:
            tee(text, f)

        with open(output_file, 'r') as f:
            result = f.read()
        assert result == 'hello_world\n'
        os.remove(output_file)
