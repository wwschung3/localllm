# app/rag/vector_store_manager.py
"""
向量儲存庫管理模組

本模組使用 faiss‑cpu 建立、存取、搜尋向量索引，並同步管理
metadata（片段文字對應表）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from tqdm import tqdm


class VectorStoreManager:
    """
    向量儲存庫管理類別
    """

    def __init__(
        self,
        index_path: str | Path = "vector_store/faiss.index",
        metadata_path: str | Path = "vector_store/metadata.json",
    ) -> None:
        """
        建構子

        :param index_path: FAISS 索引檔案路徑
        :param metadata_path: metadata JSON 檔案路徑
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: faiss.IndexFlatIP | None = None  # Inner‑Product (cosine) 索引
        self.metadata: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # 1. 初始化 / 讀取索引
    # ------------------------------------------------------------------
    def init_vector_store(self, dim: int = 512) -> None:
        """
        讀取已存在的索引；若不存在則建立新索引

        :param dim: 向量維度（預設 512）
        """
        # 確保資料夾存在
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            # 若索引是 ID‑based，請自行調整
            print(f"[向量庫] 讀取索引成功：{self.index_path}")
        else:
            # 建立一個簡單的 Inner‑Product (cosine) 索引
            self.index = faiss.IndexFlatIP(dim)
            print("[向量庫] 建立新索引")

        # 讀 metadata
        self.load_metadata()

    # ------------------------------------------------------------------
    # 2. 讀寫 metadata
    # ------------------------------------------------------------------
    def load_metadata(self) -> None:
        """
        從磁碟載入 metadata（ID → 文字）
        """
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                # json 讀回來的是 str → str，轉成 int → str
                self.metadata = {int(k): v for k, v in self.metadata.items()}
            print(f"[metadata] 載入 {len(self.metadata)} 個片段")
        else:
            self.metadata = {}
            print("[metadata] 尚未存在，從頭開始")

    def save_metadata(self) -> None:
        """
        將 metadata 寫回磁碟
        """
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print("[metadata] 儲存完成")

    # ------------------------------------------------------------------
    # 3. 增加文件（向量 + 文字）
    # ------------------------------------------------------------------
    def add_document(
        self,
        doc_id: int,
        vector: np.ndarray,
        text: str,
    ) -> None:
        """
        把單一文件（向量 + 文字）加入索引

        :param doc_id: 文件 ID（必須唯一）
        :param vector: 512‑維向量
        :param text: 文字片段
        """
        if self.index is None:
            raise RuntimeError("索引尚未初始化，請先呼叫 init_vector_store()")

        # 1️⃣ 將向量加入索引
        self.index.add(vector.reshape(1, -1))

        # 2️⃣ 儲存 metadata
        self.metadata[doc_id] = text

    # ------------------------------------------------------------------
    # 4. 搜尋
    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        根據查詢向量搜尋最相近的 k 個文件

        :param query_vector: 查詢向量
        :param k: 取前 k 個
        :return: [(doc_id, 相似度), ...]
        """
        if self.index is None:
            raise RuntimeError("索引尚未初始化，請先呼叫 init_vector_store()")

        distances, indices = self.index.search(
            query_vector.reshape(1, -1), k
        )  # distances: (1, k), indices: (1, k)

        results: List[Tuple[int, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS 會回傳 -1 代表無資料
                continue
            results.append((int(idx), float(dist)))
        return results

    # ------------------------------------------------------------------
    # 5. 清除索引
    # ------------------------------------------------------------------
    def clear_index(self) -> None:
        """
        清除現有的 FAISS 索引和 metadata，並刪除磁碟上的檔案。
        """
        if self.index is not None:
            self.index.reset()  # Reset the FAISS index
            self.index = None
            print("[向量庫] 索引已重置。")
        else:
            print("[向量庫] 索引尚未初始化，無需重置。")

        self.metadata = {}  # Clear the in-memory metadata
        print("[metadata] metadata 已清除。")

        # 刪除磁碟上的索引檔案
        if self.index_path.exists():
            os.remove(self.index_path)
            print(f"[檔案] 已刪除索引檔案：{self.index_path}")
        else:
            print(f"[檔案] 索引檔案不存在：{self.index_path}")

        # 刪除磁碟上的 metadata 檔案
        if self.metadata_path.exists():
            os.remove(self.metadata_path)
            print(f"[檔案] 已刪除 metadata 檔案：{self.metadata_path}")
        else:
            print(f"[檔案] metadata 檔案不存在：{self.metadata_path}")


# 單例實例（供其他模組直接 import）
vector_store_manager = VectorStoreManager()