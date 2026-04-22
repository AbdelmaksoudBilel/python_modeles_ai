import json
import uuid
from pathlib import Path

INPUT_ARTICLES = "data/scraped_articles.json"
INPUT_PAGES = "data/extracted_pages.json"

OUTPUT = "data/rag/chunk/rag_chunks.json"

CHUNK_SIZE = 120
OVERLAP = 30


# CHUNK TEXT

def chunk_text(text):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        start += CHUNK_SIZE - OVERLAP

    return chunks


# ARTICLES WEB

def process_articles():

    with open(INPUT_ARTICLES, "r", encoding="utf-8") as f:
        articles = json.load(f)

    chunks = []

    for art in articles:

        text_chunks = chunk_text(art["text"])

        for i, chunk in enumerate(text_chunks):

            chunks.append({
                "chunk_id": str(uuid.uuid4())[:8],
                "doc_id": art["doc_id"],
                "source_nom": art["source_nom"],
                "source_type": art["source_type"],
                "trouble": art["trouble"],
                "categorie": art["categorie"],
                "langue": art["langue"],
                "page": None,
                "chunk_index": i,
                "text": chunk,
                "word_count": len(chunk.split())
            })

    return chunks


# PDF

def process_pdfs():

    with open(INPUT_PAGES, "r", encoding="utf-8") as f:
        pdfs = json.load(f)

    chunks = []

    for pdf in pdfs:

        doc_id = pdf["source_nom"]

        for page in pdf["pages"]:

            page_chunks = chunk_text(page["text"])

            for i, chunk in enumerate(page_chunks):

                chunks.append({
                    "chunk_id": str(uuid.uuid4())[:8],
                    "doc_id": doc_id,
                    "source_nom": pdf["source_nom"],
                    "source_type": "pdf",
                    "trouble": None,
                    "categorie": None,
                    "langue": page["langue"],
                    "page": page["page"],
                    "chunk_index": i,
                    "text": chunk,
                    "word_count": len(chunk.split())
                })

    return chunks


# MAIN

def run():

    print("Chunking RAG dataset...\n")

    article_chunks = process_articles()
    pdf_chunks = process_pdfs()

    all_chunks = article_chunks + pdf_chunks

    print("Articles chunks :", len(article_chunks))
    print("PDF chunks :", len(pdf_chunks))
    print("TOTAL chunks :", len(all_chunks))

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\nDataset sauvegardé :", OUTPUT)


if __name__ == "__main__":
    run()