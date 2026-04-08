


Run these commands in your **Terminal** (not in Jupyter):

```bash
pip install textblob

# Important: Download required data (NLTK corpora)
python -m textblob.download_corpora
```

If you're using Jupyter Notebook, run this in a cell first:

```python
!pip install textblob
!python -m textblob.download_corpora
```

Then restart the kernel (Kernel → Restart) and try importing again.

### Working Code with TextBlob (After Installation)

```python
from textblob import TextBlob

def check_grammar_textblob(text):
    blob = TextBlob(text)
    corrected = blob.correct()   # Fixes spelling + basic grammar
    
    print("Original Text  :", text)
    print("Corrected Text :", corrected)
    print("Any changes?   :", str(corrected) != text)
    
    return str(corrected)

# Test it
text = "This sentence have grammer error. I goes to school yesterday and she don't like it."
check_grammar_textblob(text)
```

### 2. Even Simpler Option: pyspellchecker (Only Spelling – Very Reliable)

**Install first:**
```bash
pip install pyspellchecker
```

```python
from spellchecker import SpellChecker

def spell_checker(text):
    spell = SpellChecker()
    words = text.split()
    
    misspelled = spell.unknown(words)
    
    print("Original:", text)
    
    if not misspelled:
        print("✅ No spelling mistakes detected!")
        return text
    
    print("\nSpelling Issues Found:")
    for word in misspelled:
        correction = spell.correction(word)
        print(f"• '{word}' → Suggested: '{correction}'")
    
    # Auto-correct the whole text
    corrected = " ".join([spell.correction(word) or word for word in words])
    print("\nFully Corrected:", corrected)
    return corrected

# Test
spell_checker("I goes to shcool yesterday and buyed some fruts.")
```

### 3. Combined Spelling + Basic Correction (Best Lightweight Option)

```python
from textblob import TextBlob
from spellchecker import SpellChecker

def full_check(text):
    # Step 1: Fix spelling
    spell = SpellChecker()
    words = text.split()
    corrected_words = [spell.correction(w) or w for w in words]
    spell_fixed = " ".join(corrected_words)
    
    # Step 2: TextBlob correction
    blob = TextBlob(spell_fixed)
    final = blob.correct()
    
    print("Original       :", text)
    print("After Spelling :", spell_fixed)
    print("Final Corrected:", final)
    return str(final)

full_check("She go to market and buyed many apple. I is very happy today.")
```

### 4. AI-Powered Grammar Correction (More Accurate)

If you want better grammar fixing (not just spelling), try this **T5-based** model:

**Install:**
```bash
pip install happytransformer
```

```python
from happytransformer import HappyTextToText, TTSettings

def ai_grammar_fix(text):
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    
    settings = TTSettings(num_beams=5, min_length=1, max_length=200)
    
    result = happy_tt.generate_text("grammar: " + text, args=settings)
    
    print("Original :", text)
    print("Corrected:", result.text)
    return result.text

# First run will download the model (~800MB), so be patient
ai_grammar_fix("This is a test with many grammer mistake. I goes there yesterday.")
```

----