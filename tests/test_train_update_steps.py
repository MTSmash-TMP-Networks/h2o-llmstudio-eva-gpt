import ast
from pathlib import Path


def _load_is_optimizer_update_step():
    source = Path("llm_studio/train.py").read_text()
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "is_optimizer_update_step"
    )
    namespace = {}
    ast.fix_missing_locations(function)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<ast>", "exec"),
        namespace,
    )
    return namespace["is_optimizer_update_step"]


def test_optimizer_update_steps_include_last_partial_accumulation_block():
    is_optimizer_update_step = _load_is_optimizer_update_step()

    assert sum(is_optimizer_update_step(itr, 5, 2) for itr in range(5)) == 3
    assert sum(is_optimizer_update_step(itr, 4, 2) for itr in range(4)) == 2
    assert sum(is_optimizer_update_step(itr, 1, 4) for itr in range(1)) == 1
