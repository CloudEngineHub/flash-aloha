import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def decorator_names(function: ast.FunctionDef) -> set[str]:
    return {ast.unparse(decorator) for decorator in function.decorator_list}


def cuda_tests_without_gpu_mark() -> list[str]:
    unmarked: list[str] = []
    for path in sorted(TESTS.glob('test_*.py')):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            names = decorator_names(node)
            if 'cuda' in names and 'pytest.mark.gpu' not in names:
                unmarked.append(f'{path.name}::{node.name}')
    return unmarked


def test_every_cuda_test_is_marked_gpu() -> None:
    assert cuda_tests_without_gpu_mark() == []
