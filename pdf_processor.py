from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import io

def extract_text(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""
