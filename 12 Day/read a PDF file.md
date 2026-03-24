# read a PDF file in Python

### 1. Best & Easiest Library: **pypdf** (pure Python, actively maintained)

```bash
pip install pypdf
```

#### Basic Example – Read text from all pages

```python
from pypdf import PdfReader

# Open the PDF file
reader = PdfReader("your_file.pdf")   # replace with your PDF path

# Print number of pages
print(f"Number of pages: {len(reader.pages)}")

# Extract and print text from all pages
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n\n"

print(text)   # or save to a .txt file
```

#### Save extracted text to a file

```python
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)
```

#### Read specific page

```python
page = reader.pages[0]          # 0 = first page
print(page.extract_text())
```

### Other Popular Libraries (When to Use Them)

| Library          | Best For                          | Installation                  | Pros                              | Cons                          |
|------------------|-----------------------------------|-------------------------------|-----------------------------------|-------------------------------|
| **pypdf**        | Simple text extraction            | `pip install pypdf`           | Pure Python, fast, reliable       | Basic layout handling         |
| **pdfplumber**   | Tables + better layout            | `pip install pdfplumber`      | Excellent for tables              | Slightly slower               |
| **PyMuPDF** (fitz) | Speed + images + advanced features | `pip install pymupdf`         | Very fast, great quality          | Needs binary dependencies     |
| **pdfminer.six** | Precise character-level control   | `pip install pdfminer.six`    | Very accurate                     | More complex API              |

#### Example with **pdfplumber** (great for tables)

```python
import pdfplumber

with pdfplumber.open("your_file.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        print(text)
        
        # Extract tables
        tables = page.extract_tables()
        for table in tables:
            print(table)   # list of lists → easy to convert to pandas DataFrame
```

### For Scanned PDFs (images, not selectable text)

You need **OCR**:

```bash
pip install pymupdf pytesseract
# Also install Tesseract OCR on your system
```

Then use `pymupdf` + `pytesseract` or libraries like `unstructured` / `pdf2image`.

### Quick one-liner for many files

```python
from pathlib import Path
from pypdf import PdfReader

for pdf_file in Path(".").glob("*.pdf"):
    reader = PdfReader(pdf_file)
    full_text = "\n".join(page.extract_text() for page in reader.pages)
    print(f"--- {pdf_file.name} ---\n{full_text[:500]}...\n")
```

