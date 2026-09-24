import pytest
from backend.app.indexer.chunker import chunk_code, chunk_docs

def test_chunk_code():
    code = """
def example_function():
    print("Hello, World!")

def another_function():
    return 42
"""
    expected_chunks = [
        ("def example_function():", 1, 1),
        ("    print(\"Hello, World!\")", 2, 2),
        ("", 3, 3),
        ("def another_function():", 4, 4),
        ("    return 42", 5, 5),
    ]
    chunks = chunk_code(code)
    assert len(chunks) == len(expected_chunks)
    for (chunk, start, end), (expected_chunk, expected_start, expected_end) in zip(chunks, expected_chunks):
        assert chunk == expected_chunk
        assert start == expected_start
        assert end == expected_end

def test_chunk_docs():
    docs = """
# Title

This is a sample documentation.

## Subsection

More details here.
"""
    expected_chunks = [
        ("# Title", 1, 1),
        ("This is a sample documentation.", 2, 2),
        ("## Subsection", 4, 4),
        ("More details here.", 5, 5),
    ]
    chunks = chunk_docs(docs)
    assert len(chunks) == len(expected_chunks)
    for (chunk, start, end), (expected_chunk, expected_start, expected_end) in zip(chunks, expected_chunks):
        assert chunk == expected_chunk
        assert start == expected_start
        assert end == expected_end