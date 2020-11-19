import numpy as np
import pytest


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_indexing(document_store):
    documents = [
        {"name": "name_1", "text": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_2", "text": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_3", "text": "text_3", "embedding": np.random.rand(768).astype(np.float32)},
    ]

    document_store.write_documents(documents)
    documents_indexed = document_store.get_all_documents()

    # test if correct vector_ids are assigned
    for i, doc in enumerate(documents_indexed):
        assert doc.meta["vector_id"] == str(i)

    # test insertion of documents in an existing index fails
    with pytest.raises(Exception):
        document_store.write_documents(documents)

    # test saving the index
    document_store.save("haystack_test_faiss")

    # test loading the index
    document_store.load(sql_url="sqlite:///haystack_test.db", faiss_file_path="haystack_test_faiss")