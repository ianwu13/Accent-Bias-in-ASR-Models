# Imports Here


def preprocess_transcription(raw_text: str) -> str:
    # Remove punctuation
    for c in string.punctuation:
        raw_text = raw_text.replace(c, '')
    
    # Convert to lower case
    raw_text = raw_text.lower()
    
    return raw_text


# TODO: HERE DOWN
def plotting_function_maybe():
    pass


def correlation_function_maybe():
    pass


def make_latex_table_function_maybe():
    pass
