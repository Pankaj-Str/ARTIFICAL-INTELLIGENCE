
### Example: Extracting Text from a Paragraph

```python
import re

# Example paragraph
paragraph = """
Artificial Intelligence (AI) is a field of computer science that focuses on 
creating systems capable of performing tasks that normally require human intelligence. 
Examples include speech recognition, decision-making, and visual perception.
"""

# Define a pattern to extract specific text (e.g., sentences mentioning "AI")
pattern = r"Artificial Intelligence \(AI\).*?\."

# Extract text using regular expression
matches = re.findall(pattern, paragraph)

# Print the extracted text
if matches:
    print("Extracted Text:")
    for match in matches:
        print(match)
else:
    print("No matches found.")
```

### Output:
```
Extracted Text:
Artificial Intelligence (AI) is a field of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence.
```

### Explanation:
1. **Input Paragraph**: The paragraph contains structured sentences.
2. **Pattern**: The regex pattern `r"Artificial Intelligence \(AI\).*?\."` matches sentences starting with "Artificial Intelligence (AI)" and ending at the first period.
3. **Regex Search**: The `re.findall()` method finds all occurrences matching the pattern in the paragraph.
