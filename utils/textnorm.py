import unicodedata

def normalize_nfc(s: str) -> str:
    """
    Приводит строку к Unicode NFC (склеивает комбинируемые знаки: 'и' + '̆' -> 'й').
    Применять в препроцессе и в API перед токенизацией.
    """
    if s is None:
        return ""
    return unicodedata.normalize("NFC", str(s))