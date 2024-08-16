import pandas as pd
import pytest
from haystack import Document
from src.helpers.common import haystack_doc_to_string


def test_haystack_doc_to_string():
    doc = Document(content="Hello World!")
    assert haystack_doc_to_string(doc) == "Hello World!"
    doc = Document(dataframe=pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert (
        haystack_doc_to_string(doc)
        == "|    |   a |   b |\n|---:|----:|----:|\n|  0 |   1 |   3 |\n|  1 |   2 |   4 |"
    )
    doc = Document()
    with pytest.raises(ValueError):
        haystack_doc_to_string(doc)
