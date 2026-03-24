# QR Code Generator

### 1. Best Library: `qrcode` (with Pillow)

```bash
pip install qrcode[pil]
```

#### Simple QR Code Generator (One-liner)

```python
import qrcode

# Data you want to encode (URL, text, WiFi, etc.)
data = "https://codeswithpankaj.com"   # Change this to anything

# Generate and save QR code
img = qrcode.make(data)
img.save("my_qrcode.png")

print("QR Code generated successfully!")
```

This creates a basic black & white QR code.

### 2. Advanced QR Code with Customization

```python
import qrcode
from qrcode.constants import ERROR_CORRECT_H   # High error correction (good for logos)

# Create QR code object with settings
qr = qrcode.QRCode(
    version=1,                     # 1 to 40 (size), None = auto
    error_correction=ERROR_CORRECT_H,  # L, M, Q, H (H best with logo)
    box_size=10,                   # Size of each box in pixels
    border=4,                      # Border thickness (min 4)
)

qr.add_data("https://www.codeswithpankaj.com")   # Your data
qr.make(fit=True)                        # Auto size

# Generate image with colors
img = qr.make_image(
    fill_color="black",      # QR modules color
    back_color="white"       # Background color
)

img.save("colored_qrcode.png")
```

### 3. Beautiful QR Code with Logo in Center (Most Requested)

```python
import qrcode
from PIL import Image

def create_qr_with_logo(data, logo_path, output_path="qr_with_logo.png"):
    # High error correction for logo
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H
    )
    
    qr.add_data(data)
    qr.make(fit=True)

    # Create QR image
    qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    
    # Open logo
    logo = Image.open(logo_path)
    
    # Resize logo (about 20-25% of QR size)
    logo_size = int(qr_img.size[0] * 0.25)
    logo = logo.resize((logo_size, logo_size), Image.LANCZOS)
    
    # Paste logo in center
    pos = ((qr_img.size[0] - logo.size[0]) // 2,
           (qr_img.size[1] - logo.size[1]) // 2)
    qr_img.paste(logo, pos, mask=logo if logo.mode == 'RGBA' else None)
    
    qr_img.save(output_path)
    print(f"QR Code with logo saved as {output_path}")

# Usage
create_qr_with_logo(
    data="https://codeswithpankaj.com",
    logo_path="your_logo.png",     # Path to your logo image
    output_path="qr_with_logo.png"
)
```

### 4. Even More Beautiful Options (Recommended in 2026)

Many people now prefer **Segno** for colorful and artistic QR codes:

```bash
pip install segno pillow
```

```python
import segno

qr = segno.make_qr("Hello, World!")   # or make("https://...")

# Save with custom colors
qr.save(
    "beautiful_qr.png",
    scale=10,                    # Size
    dark="#000000",              # QR modules
    light="#FFFFFF",             # Background
    border=4,
    # You can also add gradient, rounded modules, etc. with qrcode-artistic
)
```

For **artistic QR codes** (rounded modules, gradients, animated GIFs):

```bash
pip install qrcode-artistic pillow
```

