import logging

from generator.symbol import SymbolConfiguration, GenericSymbol, SymbolGenerator

logger = logging.getLogger(__name__)


class TestCenterTextBoxManager:
    def test_get_config(self):
        c = SymbolConfiguration()
        symbol = GenericSymbol(name="STJM003", x=0, y=0)

        text_config = c.get_config(symbol)
        assert text_config.max_lines == 2


class TestSymbolGenerator:
    def test_reposition_inside_visible(self):
        c = SymbolConfiguration()
        sg = SymbolGenerator(c, (500, 500))
        symbol = GenericSymbol(
            name="STJM003", x=400, y=0, orientation=90, size_h=100, size_w=20
        )
        sg.reposition_inside_visible(symbol)
        assert symbol.x == 360
        symbol = GenericSymbol(
            name="STJM003", x=150, y=200, orientation=90, size_h=10, size_w=200
        )
        sg.reposition_inside_visible(symbol)
        assert symbol.y == 40

    def test_recalculate_positions(self):
        c = SymbolConfiguration()
        sg = SymbolGenerator(c, (500, 500))
        symbol = GenericSymbol(
            name="STJM003", x=150, y=200, size_h=10, size_w=200, orientation=90
        )
        sg.recalculate_positions(symbol, sg.ASSEMBLY_IMAGE_OFFSET)

        assert (symbol.x, symbol.y) == (190, 360)
