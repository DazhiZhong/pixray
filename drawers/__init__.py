from .vqgan import VqganDrawer
from .vdiff import VdiffDrawer
from .fftdrawer import FftDrawer
import_diffvg_drawers = False
try:
    import pydiffvg
    import diffvg
except Exception as e:
    import_diffvg_drawers = True
if import_diffvg_drawers:
    from .clipdrawer import ClipDrawer
    from .pixeldrawer import PixelDrawer
    from .dotdrawer import DotDrawer
    from .linedrawer import LineDrawer
else:
    print("no diffvg support")