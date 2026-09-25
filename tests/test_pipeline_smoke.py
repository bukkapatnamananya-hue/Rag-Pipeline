import pytest


def test_pipeline_dependencies_import():
    import app.database.models
    import app.embeddings.model

    assert app.database.models.KnowledgeChunk is not None
    assert callable(app.embeddings.model.create_embedding)
