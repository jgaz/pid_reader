import logging
from metadata import SymbolStorage

logger = logging.getLogger(__name__)


class TestSymbolStorage:
    def test_get_dataframe(self):
        ss = SymbolStorage()
        df = ss.get_dataframe()
        assert len(df.name) > 0
