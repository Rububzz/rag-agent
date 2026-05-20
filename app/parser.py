import io

import pypdf


def extract_text(file_byte: bytes, filename: str) -> str:
    if filename.lower().endswith(".txt"):
        text = file_byte.decode("utf-8")
    elif filename.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(file_byte))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    else:
        raise ValueError(
            f"Unsupported file type: {filename}. Only .txt and .pdf are supported."
        )
    return text
