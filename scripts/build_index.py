"""
Build embeddings index for the RAG corpus.

Run locally once (or whenever rag/ files change):
    pip install -r scripts/requirements.txt
    python scripts/build_index.py

Outputs: data/embeddings.json (committed and shipped to Vercel)
"""
import json
import os
import re
from pathlib import Path

import nbformat
from docx import Document
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "rag"
OUT_PATH = ROOT / "data" / "embeddings.json"

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_MAX_CHARS = 1500
OVERLAP = 200

# Per-source chunk size overrides:
#   - Manuscript: smaller chunks → finer-grained matching for specific numbers/methods
#   - faq.md: ~one Q/A pair per chunk for direct FAQ retrieval
MAX_CHARS_BY_NAME = {
    "Manuscript.docx": 1000,
    "faq.md": 600,
}

# Files in rag/ that should be excluded from the index
SKIP_FILES = {"_manuscript_extract.txt"}


def read_docx(path: Path) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_ipynb(path: Path) -> str:
    nb = nbformat.read(path, as_version=4)
    parts = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            parts.append(cell.source)
        elif cell.cell_type == "code":
            parts.append(f"```python\n{cell.source}\n```")
    return "\n\n".join(parts)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, source: str, max_chars: int) -> list[dict]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[dict] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append({"source": source, "text": current.strip()})
        current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            flush()
            step = max_chars - OVERLAP
            for i in range(0, len(para), step):
                piece = para[i : i + max_chars]
                if piece.strip():
                    chunks.append({"source": source, "text": piece.strip()})
            continue
        if len(current) + len(para) > max_chars and current:
            chunks.append({"source": source, "text": current.strip()})
            tail = current[-OVERLAP:] if len(current) > OVERLAP else current
            current = tail + "\n\n" + para
        else:
            current = (current + "\n\n" + para) if current else para
    flush()
    return chunks


def load_documents() -> list[dict]:
    all_chunks: list[dict] = []
    for path in sorted(RAG_DIR.iterdir()):
        if path.name in SKIP_FILES or path.name.startswith("_"):
            print(f"  skip (excluded): {path.name}")
            continue
        if path.suffix == ".docx":
            text = read_docx(path)
        elif path.suffix == ".ipynb":
            text = read_ipynb(path)
        elif path.suffix in (".py", ".html", ".txt", ".md"):
            text = read_text(path)
        else:
            print(f"  skip (unsupported): {path.name}")
            continue
        max_chars = MAX_CHARS_BY_NAME.get(path.name, DEFAULT_MAX_CHARS)
        chunks = chunk_text(text, path.name, max_chars)
        print(f"  {path.name}: {len(chunks)} chunks (max_chars={max_chars})")
        all_chunks.extend(chunks)
    return all_chunks


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        env_path = ROOT / ".env.local"
        if not env_path.exists():
            env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
                    break
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (env or .env.local)")

    client = OpenAI()

    print(f"Reading from {RAG_DIR}")
    chunks = load_documents()
    print(f"Total chunks: {len(chunks)}")

    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=[c["text"] for c in batch],
        )
        for c, item in zip(batch, resp.data):
            c["embedding"] = item.embedding
        print(f"  embedded {i + len(batch)}/{len(chunks)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
