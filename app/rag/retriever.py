# app/rag/retriever.py
"""
檢索器模組

本模組負責把使用者提問轉成向量，並從向量儲存庫中取得最相近的片段。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .embedding_model import embedding_model
from .vector_store_manager import vector_store_manager


def retrieve_chunks(
    query: str,
    k: int = 5,
) -> List[str]:
    """
    根據使用者提問取得最相關的文字片段

    :param query: 使用者的問題文字
    :param k: 取前 k 個最相近的片段
    :return: 相關片段文字清單
    """
    # 1️⃣ 先把提問嵌入成向量
    query_vec = embedding_model.embed_text(query)

    # 2️⃣ 在向量儲存庫中搜尋
    hits = vector_store_manager.search(query_vec, k=k)

    # 3️⃣ 取出對應的文字
    metadata = vector_store_manager.metadata
    results: List[str] = []
    for doc_id, _score in hits:
        text = metadata.get(doc_id)
        if text is not None:
            results.append(text)

    return results