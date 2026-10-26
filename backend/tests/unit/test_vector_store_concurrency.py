from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from filelock import FileLock

from app.services.vector_store import VectorStore


def test_separate_vector_stores_serialize_shared_directory_operations(
    tmp_path,
) -> None:
    state_lock = threading.Lock()
    active_calls = 0
    maximum_active_calls = 0

    class SlowCollection:
        def upsert(self, **kwargs) -> None:
            nonlocal active_calls, maximum_active_calls
            with state_lock:
                active_calls += 1
                maximum_active_calls = max(maximum_active_calls, active_calls)
            time.sleep(0.05)
            with state_lock:
                active_calls -= 1

    lock_path = tmp_path / ".codebase-vector-store.lock"
    stores = []
    for _ in range(2):
        store = VectorStore.__new__(VectorStore)
        store._operation_lock = FileLock(str(lock_path))
        store.collection = SlowCollection()
        stores.append(store)

    def write(store: VectorStore) -> None:
        store.add_documents(
            documents=["sample"],
            ids=["sample-id"],
            metadatas=[{"repo_id": "repo-a"}],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(write, stores))

    assert maximum_active_calls == 1
