# app/rag/embedding_model.py
"""
嵌入模型管理模組

本模組負責載入 sentence‑transformers 的嵌入模型，並提供單一句子或多句子
的向量化函式。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    嵌入模型封裝類別
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        建構子

        :param model_name: 要載入的模型名稱
        """
        self.model_name = model_name
        self.model: SentenceTransformer | None = None

    def load(self) -> None:
        """
        載入模型（只執行一次）
        """
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """
        將單一句子或段落轉成向量

        :param text: 需要嵌入的文字
        :return: 512‑維（或模型預設維度）的向量
        """
        if self.model is None:
            raise RuntimeError("Embedding model 尚未載入，請先呼叫 load()")
        return self.model.encode(text, convert_to_numpy=True)

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        批次嵌入多個文字片段

        :param chunks: 文字片段清單
        :return: 形狀 (len(chunks), 512) 的 numpy 陣列
        """
        if self.model is None:
            raise RuntimeError("Embedding model 尚未載入，請先呼叫 load()")
        return self.model.encode(chunks, convert_to_numpy=True)


# 單例實例（供其他模組直接 import）
embedding_model = EmbeddingModel()