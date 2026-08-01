from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re

import pandas as pd


@dataclass
class KnowledgeChunk:
    source: str
    text: str
    score: int = 0


def read_knowledge_file(file_name: str, content: bytes) -> str:
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if suffix in {"txt", "md"}:
        return content.decode("utf-8", errors="ignore")
    if suffix == "csv":
        frame = pd.read_csv(BytesIO(content))
        preview = frame.head(20).to_markdown(index=False)
        columns = "、".join(str(column) for column in frame.columns)
        return f"CSV 字段：{columns}\n\n前 20 行预览：\n{preview}"
    return content.decode("utf-8", errors="ignore")


def split_knowledge_text(source: str, text: str, chunk_size: int = 500) -> list[KnowledgeChunk]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks = []
    for start in range(0, len(cleaned), chunk_size):
        chunk = cleaned[start : start + chunk_size].strip()
        if chunk:
            chunks.append(KnowledgeChunk(source=source, text=chunk))
    return chunks


def build_query_terms(target_col: str, extra_terms: str = "") -> set[str]:
    base_terms = {
        target_col.lower(),
        "趋势",
        "异常",
        "预测",
        "指标",
        "业务",
        "口径",
        "原因",
        "建议",
        "风险",
    }
    base_terms.update(term.lower() for term in re.findall(r"[\w\u4e00-\u9fff]+", extra_terms) if len(term) >= 2)
    return base_terms


def retrieve_relevant_context(
    chunks: list[KnowledgeChunk],
    target_col: str,
    extra_terms: str = "",
    top_k: int = 4,
    max_chars: int = 1800,
) -> str:
    if not chunks:
        return ""
    terms = build_query_terms(target_col, extra_terms)
    scored = []
    for chunk in chunks:
        lowered = chunk.text.lower()
        score = sum(1 for term in terms if term and term in lowered)
        scored.append(KnowledgeChunk(source=chunk.source, text=chunk.text, score=score))
    selected = sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
    parts = []
    used_chars = 0
    for chunk in selected:
        if chunk.score <= 0 and parts:
            continue
        item = f"来源：{chunk.source}\n内容：{chunk.text}"
        if used_chars + len(item) > max_chars:
            item = item[: max(0, max_chars - used_chars)]
        if item:
            parts.append(item)
            used_chars += len(item)
        if used_chars >= max_chars:
            break
    return "\n\n".join(parts)
