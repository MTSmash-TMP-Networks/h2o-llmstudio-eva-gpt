"""Dataset package initialization."""

from llm_studio.src.datasets.exact_chained_training import (
    install_exact_chained_training,
)
from llm_studio.src.datasets.fast_sliding_window import install_fast_sliding_window
from llm_studio.src.datasets.structure_aware_sliding_window import (
    install_structure_aware_sliding_window,
)
from llm_studio.src.datasets.supervised_answer_validation import (
    install_supervised_answer_validation,
)

install_fast_sliding_window()
install_structure_aware_sliding_window()
install_supervised_answer_validation()
install_exact_chained_training()

del install_exact_chained_training
del install_fast_sliding_window
del install_structure_aware_sliding_window
del install_supervised_answer_validation
