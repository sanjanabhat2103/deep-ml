import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    pattern = r'(--|[\s,.:;?_"()\'!])'
    parts = re.split(pattern, text)
    return [part.strip() for part in parts if part.strip()]