import ast
from pathlib import Path


def _load_update_step_helpers():
    source = Path("llm_studio/train.py").read_text()
    module = ast.parse(source)
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"is_optimizer_update_step", "count_optimizer_update_steps_per_epoch"}
    ]
    namespace = {}
    for function in functions:
        ast.fix_missing_locations(function)
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "<ast>", "exec"),
        namespace,
    )
    return (
        namespace["is_optimizer_update_step"],
        namespace["count_optimizer_update_steps_per_epoch"],
    )


def test_optimizer_update_steps_include_last_partial_accumulation_block():
    is_optimizer_update_step, _ = _load_update_step_helpers()

    assert sum(is_optimizer_update_step(itr, 5, 2) for itr in range(5)) == 3
    assert sum(is_optimizer_update_step(itr, 4, 2) for itr in range(4)) == 2
    assert sum(is_optimizer_update_step(itr, 1, 4) for itr in range(1)) == 1


def test_scheduler_epoch_steps_match_optimizer_update_steps():
    _, count_optimizer_update_steps_per_epoch = _load_update_step_helpers()

    assert count_optimizer_update_steps_per_epoch(5, 2) == 3
    assert count_optimizer_update_steps_per_epoch(4, 2) == 2
    assert count_optimizer_update_steps_per_epoch(1, 4) == 1
    assert count_optimizer_update_steps_per_epoch(1000, 4) == 250
