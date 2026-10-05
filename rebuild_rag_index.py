"""Rebuild the RAG index from the FULL text of every PDF.
Run on your own computer. Needs: pip install pymupdf chromadb sentence-transformers
Output: ./rag_database_new  (same collection name / embedder / metadata key as the app)
"""
import os, re, sys, fitz, chromadb
from chromadb.utils import embedding_functions

PDF_DIR = r"C:\Users\raalnuba\Desktop\Lithium_Batteries_PDFs"
OUT_DIR = "rag_database_new"
CHUNK_WORDS, OVERLAP = 600, 100

def pdf_text(path):
    doc = fitz.open(path)
    pages = [p.get_text("text") for p in doc]
    return pages

def chunks(words, size=CHUNK_WORDS, overlap=OVERLAP):
    step = size - overlap
    for i in range(0, max(len(words), 1), step):
        piece = words[i:i+size]
        if len(piece) >= 50 or i == 0:
            yield " ".join(piece)
        if i + size >= len(words): break

emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=OUT_DIR)
try: client.delete_collection("solid_electrolytes")
except Exception: pass
col = client.create_collection("solid_electrolytes", embedding_function=emb,
                               metadata={"hnsw:space": "cosine"})

report, ids, docs, metas = [], [], [], []
files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
for fn in files:
    try:
        pages = pdf_text(os.path.join(PDF_DIR, fn))
    except Exception as e:
        report.append((fn, 0, 0, f"ERROR {e}")); continue
    text = re.sub(r"\s+", " ", " ".join(pages)).strip()
    words = text.split(" ")
    report.append((fn, len(pages), len(text), "OK" if len(pages) > 1 and len(text) > 5000 else "CHECK"))
    for k, c in enumerate(chunks(words)):
        ids.append(f"{fn}::{k}"); docs.append(c); metas.append({"source": fn, "chunk": k})
for i in range(0, len(ids), 500):
    col.add(ids=ids[i:i+500], documents=docs[i:i+500], metadatas=metas[i:i+500])

print(f"{len(files)} PDFs, {len(ids)} chunks")
bad = [r for r in report if r[3] != "OK"]
print("PDFs needing a look (1 page, little text, or errors):")
for r in bad: print("  ", r)
import csv
with open("rag_rebuild_report.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["file","pages","chars","status"]); w.writerows(report)
