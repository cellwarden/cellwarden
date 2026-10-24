"""scpipe is now CellWarden: ``import scpipe`` and ``import scpipe.<module>`` give ``cellwarden``.

Kept for one release so code written for scpipe keeps working; use ``cellwarden`` instead.
"""

import sys
import warnings

import cellwarden

warnings.warn("scpipe has been renamed CellWarden: import cellwarden instead", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = cellwarden
