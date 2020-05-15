import logging

import pandas as pd

from metadata import SymbolStorage
from symbol import SymbolConfiguration, GenericSymbol

logger = logging.getLogger(__name__)


class TestCenterTextBoxManager:
    def test_get_config(self):
        c = SymbolConfiguration()
        symbol = GenericSymbol(name="STJM003", x=0, y=0)

        text_config = c.get_config(symbol)
        assert text_config.max_lines == 2
