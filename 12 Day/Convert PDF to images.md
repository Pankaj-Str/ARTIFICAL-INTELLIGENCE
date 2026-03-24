# Convert PDF to images

### 1. Recommended: **PyMuPDF** (fitz) – Fastest, no external dependencies, excellent quality

```bash
pip install pymupdf pillow   # Pillow is optional but helpful
```

#### Convert PDF to high-quality PNG images (one per page)

```python
import fitz  # PyMuPDF
import os

def pdf_to_images(pdf_path, output_folder="pdf_images", dpi=300):
    os.makedirs(output_folder, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    print(f"Converting {len(doc)} pages to images...")
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Higher dpi = better quality (300 is good, 600 for very high quality)
        pix = page.get_pixmap(dpi=dpi, alpha=True)   # alpha=True for transparency in PNG
        
        output_path = os.path.join(output_folder, f"page_{page_num+1:03d}.png")
        pix.save(output_path)
        
        print(f"Saved: {output_path}")
    
    doc.close()
    print("Conversion completed!")

# Usage
pdf_to_images("your_document.pdf")
```

#### For JPG instead of PNG (smaller file size)

```python
pix = page.get_pixmap(dpi=300)
pix.save(output_path, jpg_quality=95)   # 0-100 quality
```

### 2. Alternative: **pdf2image** (very simple, uses Poppler)

This is popular and easy but requires installing **Poppler** on your system.

```bash
pip install pdf2image
```

**Installation of Poppler:**

- **Windows**: Download from https://github.com/oschwartz10612/poppler-windows/releases/ → extract and add `bin/` to PATH
- **Mac**: `brew install poppler`
- **Linux**: `sudo apt install poppler-utils`

#### Code

```python
from pdf2image import convert_from_path
import os

def pdf_to_images_pdf2image(pdf_path, output_folder="pdf_images", dpi=300):
    os.makedirs(output_folder, exist_ok=True)
    
    images = convert_from_path(pdf_path, dpi=dpi)
    
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"page_{i+1:03d}.png")
        image.save(output_path, "PNG")
        print(f"Saved: {output_path}")

# Usage
pdf_to_images_pdf2image("your_document.pdf")
```

### Comparison (2026)

| Feature                  | PyMuPDF (fitz)          | pdf2image               |
|--------------------------|-------------------------|-------------------------|
| Speed                    | Very Fast               | Fast                    |
| External dependencies    | None (pure Python wheel)| Poppler required        |
| Quality control          | Excellent (dpi, zoom)   | Good (dpi)              |
| File size / Transparency | Great                   | Good                    |
| Best for                 | Most users, production  | Quick scripts           |

**Recommendation**: Use **PyMuPDF** unless you already have Poppler installed and prefer the simplest API.

### Extra Options You Might Need

**Convert only specific pages** (e.g., pages 1 and 5):

```python
# PyMuPDF
for page_num in [0, 4]:        # 0-based index
    page = doc.load_page(page_num)
    ...
```

**Higher quality / custom size**:

```python
# PyMuPDF - zoom factor instead of dpi
mat = fitz.Matrix(4, 4)        # 4x zoom ≈ 600 dpi
pix = page.get_pixmap(matrix=mat)
```

**Convert from bytes** (no file on disk):

```python
with open("file.pdf", "rb") as f:
    pdf_bytes = f.read()

doc = fitz.open(stream=pdf_bytes, filetype="pdf")
```
