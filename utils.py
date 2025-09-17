import pdfplumber
import pandas as pd
import re
from typing import Dict, Any, List, Union
from pdf2image import convert_from_path
import pytesseract
import tempfile, os

# -------------------------------
# OCR helper
# -------------------------------
def ocr_page_as_text(file_bytes, page_number: int) -> str:
    """Run OCR on a single page of a PDF (uploaded BytesIO)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes.getbuffer())
        tmp_path = tmp.name
    images = convert_from_path(tmp_path, first_page=page_number + 1, last_page=page_number + 1)
    os.remove(tmp_path)
    return pytesseract.image_to_string(images[0])

# -------------------------------
# File Validation
# -------------------------------
def validate_file(file):
    """Check if uploaded file has a supported extension (pdf, xlsx, xls)."""
    allowed_extensions = (".pdf", ".xlsx", ".xls")
    if file is None:
        return False, "No file uploaded."

    name = file.name if hasattr(file, "name") else str(file)
    if name.lower().endswith(allowed_extensions):
        return True, "File is valid."
    return False, "Unsupported file type. Please upload PDF or Excel."

# -------------------------------
# PDF Extraction with OCR fallback
# -------------------------------
def extract_pdf_content(file) -> Dict[str, Any]:
    content = {"type": "pdf", "text": []}
    try:
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text() or ""
                if txt.count("�") > 2 or len(re.findall(r"\d", txt)) < 2:
                    txt = ocr_page_as_text(file, i)
                content["text"].append(txt)
        content["pages"] = len(content["text"])
    except Exception as e:
        content["error"] = f"Error reading PDF: {e}"
    return content

# -------------------------------
# Excel Extraction
# -------------------------------
def extract_excel_content(file) -> Dict[str, Any]:
    content = {"type": "excel", "sheets": {}}
    try:
        xls = pd.ExcelFile(file)
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(file, sheet_name=sheet_name)
                content["sheets"][sheet_name] = df
            except Exception as e:
                content["sheets"][sheet_name] = f"Error reading sheet {sheet_name}: {e}"
        content.setdefault("metadata", {})["sheet_count"] = len(content["sheets"])
    except Exception as e:
        content["error"] = f"Error reading Excel: {e}"
    return content

# -------------------------------
# Keyword Extraction
# -------------------------------
def extract_financial_keywords(text: str) -> List[str]:
    keywords = [
        "revenue", "income", "expenses", "profit", "loss", "assets", "liabilities",
        "equity", "cash flow", "operating", "net", "gross", "balance", "dividend",
        "depreciation", "amortization"
    ]
    return [kw for kw in keywords if re.search(rf"\b{kw}\b", text, re.IGNORECASE)]

# -------------------------------
# Number Extraction
# -------------------------------
def extract_numerical_data(text: str) -> List[str]:
    pattern = r"(?:\$)?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?"
    return re.findall(pattern, text)

# -------------------------------
# Convert amounts to numbers
# -------------------------------
def parse_amount_to_number(s: str) -> Union[float, None]:
    s = s.strip().lower().replace(",", "")
    s = re.sub(r"^\$", "", s)

    m = re.match(r"([\d\.]+)\s*million", s)
    if m: return float(m.group(1)) * 1_000_000
    m = re.match(r"([\d\.]+)\s*billion", s)
    if m: return float(m.group(1)) * 1_000_000_000
    m = re.match(r"([\d\.]+)\s*%$", s)
    if m: return float(m.group(1)) / 100.0

    try:
        return float(re.findall(r"[\d\.]+", s)[0])
    except Exception:
        return None

# -------------------------------
# Build compact document context for LLM
# -------------------------------
def build_document_context(extracted_content: Dict[str, Any], max_chars: int = 3500) -> str:
    parts = []
    if extracted_content.get("type") == "pdf" and extracted_content.get("text"):
        for i, page_text in enumerate(extracted_content["text"][:4]):
            snippet = page_text.strip().replace("\n", " ")
            parts.append(f"Page {i+1}: {snippet[:800]}")
    elif extracted_content.get("type") == "excel" and extracted_content.get("sheets"):
        for sheet_name, df in list(extracted_content["sheets"].items())[:3]:
            if isinstance(df, pd.DataFrame) and not df.empty:
                parts.append(f"Sheet {sheet_name} top rows:\n{df.head(5).to_csv(index=False)}")

    if extracted_content.get("financial_keywords"):
        parts.append("Keywords: " + ", ".join(extracted_content["financial_keywords"][:12]))
    if extracted_content.get("numerical_data"):
        parts.append("Numbers found: " + ", ".join(extracted_content["numerical_data"][:12]))

    context = "\n\n".join(parts)
    return context[:max_chars] + ("\n\n[TRUNCATED]" if len(context) > max_chars else "")

# -------------------------------
# Enrich extracted content
# -------------------------------
def enrich_processed_content(content: Dict[str, Any], file=None) -> Dict[str, Any]:
    if not isinstance(content, dict):
        return content

    text_blocks = []
    if content.get("type") == "pdf":
        text_blocks = content.get("text", [])
    elif content.get("type") == "excel":
        for sheet, df in content.get("sheets", {}).items():
            if isinstance(df, pd.DataFrame):
                text_blocks.append(df.head(10).to_csv(index=False))

    full_text = "\n".join(text_blocks)
    content["financial_keywords"] = extract_financial_keywords(full_text)
    content["numerical_data"] = extract_numerical_data(full_text)

    return content
