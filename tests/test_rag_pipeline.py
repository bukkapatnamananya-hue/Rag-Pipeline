import importlib


def test_core_rag_modules_exist():
    for module_name in [
        'app.ingestion.pdf_ingestion',
        'app.search.retrieval',
        'app.rag.query_pipeline',
        'app.evaluation.metrics',
    ]:
        assert importlib.import_module(module_name) is not None


def test_retrieval_modes_are_defined():
    retrieval = importlib.import_module('app.search.retrieval')
    assert hasattr(retrieval, 'dense_retrieval')
    assert hasattr(retrieval, 'sparse_retrieval')
    assert hasattr(retrieval, 'hybrid_retrieval')
