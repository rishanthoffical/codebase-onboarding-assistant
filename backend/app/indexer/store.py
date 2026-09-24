from typing import Any, Dict, List

class VectorStore:
    def __init__(self):
        self.store = {}

    def upsert(self, chunk_id: str, metadata: Dict[str, Any]) -> None:
        self.store[chunk_id] = metadata

    def get(self, chunk_id: str) -> Dict[str, Any]:
        return self.store.get(chunk_id, {})

    def delete(self, chunk_id: str) -> None:
        if chunk_id in self.store:
            del self.store[chunk_id]

    def list(self) -> List[str]:
        return list(self.store.keys())

# Initialize the vector store
vector_store = VectorStore()