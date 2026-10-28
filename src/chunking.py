import os
import re
import json
from langchain_community.document_loaders import PyPDFLoader
from src.config import PDF_FOLDER, OUTPUT_FILE


MAX_CHUNK_WORDS = 120


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = text.replace("\uf0b7", "•")
    text = text.replace("\u0000", " ")
    text = text.replace("￾", "-")
    text = text.replace("\u00a0", " ")
    text = text.replace("ﬀ", "ff")
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    return text.strip()


def is_page_header(line):
    return bool(re.match(r"^\d{1,2}/\d{1,2}\s*\|\s*.+$", line))


def is_page_number(line):
    return bool(re.match(r"^\d{1,2}/\d{1,2}$", line))


def is_bullet(line):
    return line.strip().startswith(("•", "-", "❖", "▪", "◦"))


def normalize_bullet(line):
    return re.sub(r"^[•\-❖▪◦]\s*", "", line).strip()


def is_sentence_end(text):
    return text.strip().endswith((".", "?", "!", ":"))


def is_valid_chunk(text):
    text = clean_text(text)
    return len(text) >= 8 and len(text.split()) >= 2


def should_flush_chunk(text, max_words=MAX_CHUNK_WORDS):
    return len(text.split()) >= max_words and is_sentence_end(text)


def is_section(line):
    line = clean_text(line)

    if len(line) < 3 or len(line) > 90:
        return False

    if is_page_header(line) or is_page_number(line):
        return False

    ignored = {
        "POLICY",
        "HR ORIENTATION",
        "ATTENDANCE POLICY",
        "NAGWA",
        "THANK YOU",
        "?",
    }

    if line.upper() in ignored:
        return False

    if line.isupper():
        return True

    words = line.split()

    if len(words) <= 8 and not line.endswith("."):
        capitalized_words = sum(1 for w in words if w[:1].isupper())
        if capitalized_words >= len(words) * 0.6:
            return True

    return False


def split_numbered_rules(text):
    """
    Split text containing numbered rules:
    6. Rule text 7. Rule text 8. Rule text
    into separate chunks.
    """
    text = clean_text(text)

    if not re.search(r"\b\d+\.\s+", text):
        return []

    parts = re.split(r"(?=\b\d+\.\s+)", text)

    rules = []
    for part in parts:
        part = clean_text(part)
        if re.match(r"^\d+\.\s+", part):
            rules.append(part)

    return rules


def extract_rule_number(text):
    match = re.match(r"^(\d+)\.\s+", text.strip())
    return match.group(1) if match else None


def build_embedding_text(chunk):
    return f"""
Company HR Policy
Document: {chunk["document"]}
Section: {chunk["section"]}
Page: {chunk.get("page", "Unknown")}
{chunk["text"]}
""".strip()


def create_single_chunk(text, section, document, source, page=None, chunk_type="paragraph"):
    text = clean_text(text)

    if not is_valid_chunk(text):
        return None

    chunk = {
        "text": text,
        "section": section.title() if section else "General",
        "document": document,
        "source": os.path.basename(source),
        "page": page,
        "type": chunk_type,
    }

    rule_number = extract_rule_number(text)
    if rule_number:
        chunk["rule_number"] = rule_number

    chunk["embedding_text"] = build_embedding_text(chunk)
    return chunk


def create_chunks(text, section, document, source, page=None):
    text = clean_text(text)

    if not is_valid_chunk(text):
        return []

    numbered_rules = split_numbered_rules(text)

    # If the paragraph contains multiple numbered rules, split each rule separately.
    if len(numbered_rules) > 1:
        chunks = []

        for rule in numbered_rules:
            chunk = create_single_chunk(
                text=rule,
                section=section,
                document=document,
                source=source,
                page=page,
                chunk_type="numbered_rule",
            )

            if chunk:
                chunks.append(chunk)

        return chunks

    # Otherwise keep as a normal paragraph.
    chunk = create_single_chunk(
        text=text,
        section=section,
        document=document,
        source=source,
        page=page,
        chunk_type="paragraph",
    )

    return [chunk] if chunk else []


def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    document_name = os.path.basename(file_path).replace(".pdf", "").replace(".", " ")

    chunks = []
    current_section = "General"

    for page_num, doc in enumerate(docs, start=1):
        raw_text = doc.page_content

        lines = [
            clean_text(line)
            for line in raw_text.split("\n")
            if clean_text(line)
        ]

        if not lines:
            continue

        paragraph_buffer = ""

        def flush_buffer():
            nonlocal paragraph_buffer

            if paragraph_buffer.strip():
                new_chunks = create_chunks(
                    text=paragraph_buffer,
                    section=current_section,
                    document=document_name,
                    source=file_path,
                    page=page_num,
                )

                chunks.extend(new_chunks)
                paragraph_buffer = ""

        for line in lines:
            line = clean_text(line)

            if not line:
                continue

            if is_page_header(line) or is_page_number(line):
                continue

            if line.upper() in {
                "ATTENDANCE POLICY",
                "HR ORIENTATION",
                "NAGWA",
                "POLICY",
                "THANK YOU",
                "?",
            }:
                continue

            if is_section(line):
                flush_buffer()
                current_section = line
                continue

            if is_bullet(line):
                line = normalize_bullet(line)

            if paragraph_buffer:
                paragraph_buffer += " " + line
            else:
                paragraph_buffer = line

            if should_flush_chunk(paragraph_buffer):
                flush_buffer()

        flush_buffer()

    return chunks


def build_chunks():
    all_chunks = []

    print("📂 Reading PDFs from:", PDF_FOLDER)

    for file in os.listdir(PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, file)
            print(f"🔄 Processing: {file}")

            pdf_chunks = process_pdf(path)
            all_chunks.extend(pdf_chunks)

    print(f"\n✅ Total chunks created: {len(all_chunks)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_chunks()