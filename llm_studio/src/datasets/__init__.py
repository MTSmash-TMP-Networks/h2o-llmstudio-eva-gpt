"""Dataset package initialization."""

from llm_studio.src.datasets.fast_sliding_window import install_fast_sliding_window
from llm_studio.src.datasets.structure_aware_sliding_window import (
    install_structure_aware_sliding_window,
)

install_fast_sliding_window()
install_structure_aware_sliding_window()

del install_fast_sliding_window
del install_structure_aware_sliding_window
