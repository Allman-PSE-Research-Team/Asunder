import importlib


def test_restructured_namespace_imports():
    """Smoke-test the canonical module namespaces introduced by the refactor."""
    module_names = [
        "asunder.base.column_generation",
        "asunder.base.branch_and_price",
        "asunder.base.algorithms.community",
        "asunder.base.algorithms.modular_VFD",
        "asunder.nlbnp.algorithms.refinement",
        "asunder.nlbnp.algorithms.core_periphery",
        "asunder.nlbnp.case_studies.runner",
        "asunder.nlbnp.workflow",
    ]
    for name in module_names:
        assert importlib.import_module(name) is not None
