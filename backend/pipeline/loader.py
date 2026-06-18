import io
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
import docx as python_docx
import pytesseract

# HEIC/HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx",
                        ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".heic", ".heif"}
def load_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def load_markdown(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def load_pdf(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [page.get_text() for page in doc]
    return "\n\n".join(pages)


def load_docx(data: bytes) -> str:
    doc = python_docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_image(data: bytes) -> str:
    img = Image.open(io.BytesIO(data))
    text = pytesseract.image_to_string(img)
    return text.strip() or "[No text found in image]"


LOADERS = {
    ".txt":      load_txt,
    ".md":       load_markdown,
    ".markdown": load_markdown,
    ".pdf":      load_pdf,
    ".docx":     load_docx,
    ".png":      load_image,
    ".jpg":      load_image,
    ".jpeg":     load_image,
    ".webp":     load_image,
    ".tiff":     load_image,
    ".bmp":      load_image,
    ".heic":     load_image,
    ".heif":     load_image,
}


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return LOADERS[ext](data)
