# Imports Here


def preprocess_transcription(raw_text: str) -> str:
    # Remove punctuation
    for c in string.punctuation:
        raw_text = raw_text.replace(c, '')
    
    # Convert to lower case
    raw_text = raw_text.lower()
    
    return raw_text