
from typing import List

from db import delete_resume, delete_saved_searches_for_resume, enforce_saved_search_limit
from faiss_search_index import delete_index_dir

def enforce_limit_and_cleanup(user_id: int, keep_n: int = 3) -> List[int]:
    deleted_ids = enforce_saved_search_limit(user_id=user_id, keep_n=keep_n)
    for sid in deleted_ids:
        delete_index_dir(sid)
    return deleted_ids

def delete_resume_and_cleanup(user_id: int, resume_id: int) -> List[int]:
    # delete saved searches tied to resume (DB) and indices (disk), then delete resume
    search_ids = delete_saved_searches_for_resume(user_id=user_id, resume_id=resume_id)
    for sid in search_ids:
        delete_index_dir(sid)
    delete_resume(user_id=user_id, resume_id=resume_id)
    return search_ids
