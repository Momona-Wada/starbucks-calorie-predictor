# text_preprocessor.py
def preprocess_text(text: str) -> str:
    """A function to preprocess raw text by lowercasing
       and replacing certain synonyms."""
    text = text.lower()
    synonyms = {
        "calories": "calorie",
        "low calorie": "low_calorie",
        "low-calorie": "low_calorie"
    }
    for key, val in synonyms.items():
        text = text.replace(key, val)
    return text
