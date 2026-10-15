"""
Linguistics, Orthography, and Normalization Engine for Multi-Language TTS.
Handles:
1. Hausa hooked/glottal consonant replacement ('b -> ɓ, 'd -> ɗ, 'k -> ƙ, 'y -> ƴ).
2. Number, currency, and symbol expansion for English, French, and Hausa.
3. Educational Markdown, code, and diagram stripping.
4. Intelligent abbreviation-aware sentence splitting for streaming chunk generation.
"""

import re
from typing import List

# ==============================================================================
# 1. HAUSA ORTHOGRAPHY & HOOKED CONSONANT NORMALIZATION
# ==============================================================================

# Common Latin keyboard typing patterns for Hausa glottal/hooked letters:
# 'b / b' -> ɓ, 'd / d' -> ɗ, 'k / k' -> ƙ, 'y / y' -> ƴ
HAUSA_HOOKED_MAP = [
    # Lowercase hooked letters
    (re.compile(r"['`‘](\b[bB])|([bB])['`’]", re.IGNORECASE), lambda m: "Ɓ" if (m.group(1) or m.group(2)).isupper() else "ɓ"),
    (re.compile(r"['`‘](\b[dD])|([dD])['`’]", re.IGNORECASE), lambda m: "Ɗ" if (m.group(1) or m.group(2)).isupper() else "ɗ"),
    (re.compile(r"['`‘](\b[kK])|([kK])['`’]", re.IGNORECASE), lambda m: "Ƙ" if (m.group(1) or m.group(2)).isupper() else "ƙ"),
    (re.compile(r"['`‘](\b[yY])|([yY])['`’]", re.IGNORECASE), lambda m: "Ƴ" if (m.group(1) or m.group(2)).isupper() else "ƴ"),
]

# Explicit substitutions for common words typed with apostrophes
HAUSA_WORD_SUBSTITUTIONS = {
    "'baki": "ɓaki",
    "'barna": "ɓarna",
    "'boye": "ɓoye",
    "'dali": "ɗali",
    "'dalibi": "ɗalibi",
    "'dalibai": "ɗalibai",
    "'dakin": "ɗakin",
    "'daki": "ɗaki",
    "'dan": "ɗan",
    "'daya": "ɗaya",
    "'kasa": "ƙasa",
    "'karami": "ƙarami",
    "'kankana": "ƙanƙana",
    "'karfi": "ƙarfi",
    "'kofa": "ƙofa",
    "'yanci": "ƴanci",
    "'ya'ya": "ƴaƴa",
    "'yammaci": "ƴammaci",
    "'yar": "ƴar",
    "'yan": "ƴan",
}


def normalize_hausa_orthography(text: str) -> str:
    """
    Transforms ASCII apostrophe-typed Hausa text into accurate Unicode characters:
    - 'b / b' -> ɓ, 'B -> Ɓ
    - 'd / d' -> ɗ, 'D -> Ɗ
    - 'k / k' -> ƙ, 'K -> Ƙ
    - 'y / y' -> ƴ, 'Y -> Ƴ
    """
    if not text:
        return ""

    # 1. Exact dictionary substitutions first for precision
    words = text.split()
    normalized_words = []
    for w in words:
        lower_w = w.lower().strip(".,!?;:\"'()")
        if lower_w in HAUSA_WORD_SUBSTITUTIONS:
            repl = HAUSA_WORD_SUBSTITUTIONS[lower_w]
            # Match capitalization
            if w[0].isupper():
                repl = repl.capitalize()
            # Preserve punctuation around it
            w = re.sub(re.escape(lower_w), repl, w, flags=re.IGNORECASE)
        normalized_words.append(w)
    
    text = " ".join(normalized_words)

    # 2. General regex patterns for hooked consonants
    for pattern, repl_fn in HAUSA_HOOKED_MAP:
        text = pattern.sub(repl_fn, text)

    return text


# ==============================================================================
# 2. HAUSA NUMERAL & SYMBOL EXPANSION
# ==============================================================================

HAUSA_DIGITS = {
    0: "koma",
    1: "ɗaya",
    2: "biyu",
    3: "uku",
    4: "huɗu",
    5: "biyar",
    6: "shida",
    7: "bakwai",
    8: "takwas",
    9: "tara",
    10: "goma",
    11: "goma sha ɗaya",
    12: "goma sha biyu",
    13: "goma sha uku",
    14: "goma sha huɗu",
    15: "goma sha biyar",
    16: "goma sha shida",
    17: "goma sha bakwai",
    18: "goma sha takwas",
    19: "goma sha tara",
    20: "ashirin",
    30: "talatin",
    40: "arba'in",
    50: "hamsin",
    60: "sittin",
    70: "saba'in",
    80: "tamanin",
    90: "tasa'in",
    100: "ɗari",
    1000: "dubu",
    1000000: "miliyan",
}


def hausa_int_to_words(n: int) -> str:
    """Converts integers (0 - 999,999,999) into spoken Hausa words."""
    if n < 0:
        return f"ragi {hausa_int_to_words(-n)}"
    if n in HAUSA_DIGITS and n <= 20:
        return HAUSA_DIGITS[n]
    if n < 100:
        tens = (n // 10) * 10
        rem = n % 10
        if rem == 0:
            return HAUSA_DIGITS.get(tens, str(n))
        return f"{HAUSA_DIGITS.get(tens, '')} da {HAUSA_DIGITS.get(rem, '')}"
    if n < 1000:
        hundreds = n // 100
        rem = n % 100
        h_str = "ɗari" if hundreds == 1 else f"ɗari {HAUSA_DIGITS.get(hundreds, '')}"
        if rem == 0:
            return h_str
        return f"{h_str} da {hausa_int_to_words(rem)}"
    if n < 1000000:
        thousands = n // 1000
        rem = n % 1000
        t_str = "dubu" if thousands == 1 else f"dubu {hausa_int_to_words(thousands)}"
        if rem == 0:
            return t_str
        return f"{t_str} da {hausa_int_to_words(rem)}"
    if n < 1000000000:
        millions = n // 1000000
        rem = n % 1000000
        m_str = "miliyan ɗaya" if millions == 1 else f"miliyan {hausa_int_to_words(millions)}"
        if rem == 0:
            return m_str
        return f"{m_str} da {hausa_int_to_words(rem)}"
    return str(n)


def expand_hausa_symbols(text: str) -> str:
    """Expands numbers, percentages, currencies, and mathematical symbols in Hausa."""
    # Percentages: e.g. 50% -> kashi hamsin cikin dari
    def _pct(m):
        val = int(m.group(1))
        return f"kashi {hausa_int_to_words(val)} cikin ɗari"
    text = re.sub(r"(\d+)%", _pct, text)

    # Currencies: ₦500 -> naira dari biyar, $10 -> dala goma
    def _naira(m):
        val = int(m.group(1))
        return f"naira {hausa_int_to_words(val)}"
    text = re.sub(r"[₦N](\d+)", _naira, text)

    def _dollar(m):
        val = int(m.group(1))
        return f"dala {hausa_int_to_words(val)}"
    text = re.sub(r"\$(\d+)", _dollar, text)

    # Standalone numbers:
    def _num(m):
        try:
            return hausa_int_to_words(int(m.group(0)))
        except Exception:
            return m.group(0)
    text = re.sub(r"\b\d+\b", _num, text)

    return text


# ==============================================================================
# 3. MULTI-LANGUAGE NUMBER & SYMBOL EXPANSION
# ==============================================================================

def expand_numbers_and_symbols(text: str, lang: str = "en") -> str:
    """
    Expands digits, currencies, and percentages into written spoken words.
    Uses num2words for English and French, and native Hausa numeral expander for Hausa.
    """
    lang = (lang or "en").lower().strip()

    if lang == "ha":
        return expand_hausa_symbols(text)

    # For English / French:
    try:
        from num2words import num2words
    except ImportError:
        # Graceful fallback if num2words not yet installed
        return text

    if lang == "fr":
        # French percentages
        text = re.sub(r"(\d+)%", lambda m: f"{num2words(int(m.group(1)), lang='fr')} pour cent", text)
        # French Euro / Dollar
        text = re.sub(r"(\d+)\s*€", lambda m: f"{num2words(int(m.group(1)), lang='fr')} euros", text)
        text = re.sub(r"\$(\d+)", lambda m: f"{num2words(int(m.group(1)), lang='fr')} dollars", text)
        # Integers
        text = re.sub(r"\b\d+\b", lambda m: num2words(int(m.group(0)), lang="fr"), text)
        return text

    # Default: English
    text = re.sub(r"(\d+)%", lambda m: f"{num2words(int(m.group(1)), lang='en')} percent", text)
    text = re.sub(r"\$(\d+)", lambda m: f"{num2words(int(m.group(1)), lang='en')} dollars", text)
    text = re.sub(r"[₦N](\d+)", lambda m: f"{num2words(int(m.group(1)), lang='en')} naira", text)
    text = re.sub(r"\b\d+\b", lambda m: num2words(int(m.group(0)), lang="en"), text)
    return text


# ==============================================================================
# 4. MARKDOWN & EDUCATIONAL CONTENT SANITIZER
# ==============================================================================

def clean_markdown_for_speech(text: str) -> str:
    """
    Strips raw markdown syntax, code snippets, and diagram tags so the text sounds
    natural when read aloud by a human voice.
    """
    if not text:
        return ""

    # 1. Strip image & diagram placeholders: ![alt](DIAGRAM: ...) or ![alt](IMG: ...)
    text = re.sub(r"!\[([^\]]*)\]\((?:DIAGRAM|IMG):[^\)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", text)

    # 2. Convert markdown links [Label](url) -> Label
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # 3. Handle code blocks: ```python ... ``` -> "code example omitted"
    text = re.sub(r"```[a-zA-Z]*[\s\S]*?```", " Code snippet. ", text)
    # Inline code `variable` -> variable
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 4. Strip header markers (#, ##, ###)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # 5. Strip blockquotes >
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

    # 6. Strip list bullet markers: *, -, +, 1.
    text = re.sub(r"^\s*[\*\-\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # 7. Strip bold/italic/strikethrough markers: **text**, *text*, ~~text~~
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # 8. Collapse excessive whitespace and blank lines
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    return text


# ==============================================================================
# 5. SMART ABBREVIATION-AWARE SENTENCE CHUNKER
# ==============================================================================

# Common honorifics and abbreviations that should NOT trigger a sentence split
ABBREVIATIONS = [
    r"Dr\.", r"Prof\.", r"Mr\.", r"Mrs\.", r"Ms\.", r"e\.g\.", r"i\.e\.",
    r"vs\.", r"etc\.", r"Fig\.", r"Dept\.", r"Univ\.", r"St\.", r"No\.",
    r"Alh\.", r"Mal\.", r"Engr\."  # Common Nigerian / Northern titles
]
ABBR_PATTERN = re.compile(r"|".join(ABBREVIATIONS), re.IGNORECASE)


def split_into_sentences(text: str, max_chunk_words: int = 35) -> List[str]:
    """
    Splits text into cohesive sentences suitable for immediate audio synthesis.
    Protects abbreviations and decimal numbers from accidental splitting.
    If a sentence is overly long, splits gently on commas/semicolons/conjunctions.
    """
    if not text:
        return []

    # Temporary mask for protected abbreviations
    abbr_tokens = {}
    def _mask_abbr(m):
        token = f"__ABBR_{len(abbr_tokens)}__"
        abbr_tokens[token] = m.group(0)
        return token

    masked = ABBR_PATTERN.sub(_mask_abbr, text)

    # Mask decimals (e.g. 3.14)
    decimal_tokens = {}
    def _mask_dec(m):
        token = f"__DEC_{len(decimal_tokens)}__"
        decimal_tokens[token] = m.group(0)
        return token

    masked = re.sub(r"\b\d+\.\d+\b", _mask_dec, masked)

    # Split on sentence boundaries: (. ! ? or semicolon / newline followed by space or capital)
    raw_sentences = re.split(r"(?<=[.!?;\n])\s+", masked)

    final_sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue

        # Restore abbreviations and decimals
        for token, original in abbr_tokens.items():
            s = s.replace(token, original)
        for token, original in decimal_tokens.items():
            s = s.replace(token, original)

        # If a single sentence is excessively long (> max_chunk_words), split on sub-clauses
        words = s.split()
        if len(words) > max_chunk_words:
            clauses = re.split(r"(?<=[,;:])\s+", s)
            current_clause = []
            for c in clauses:
                current_clause.append(c)
                if len(" ".join(current_clause).split()) >= 15:
                    final_sentences.append(" ".join(current_clause).strip())
                    current_clause = []
            if current_clause:
                final_sentences.append(" ".join(current_clause).strip())
        else:
            final_sentences.append(s)

    return [s for s in final_sentences if len(s.strip()) > 1]


# ==============================================================================
# 6. UNIFIED PIPELINE FOR SPEECH PREPARATION
# ==============================================================================

def prepare_text_for_speech(raw_text: str, lang: str = "en") -> List[str]:
    """
    Full preparation pipeline:
    1. Markdown & diagram stripping.
    2. Hausa orthography normalization (if Hausa).
    3. Number and symbol word expansion.
    4. Sentence boundary chunking.
    Returns list of clean sentences ready for the voice engine.
    """
    # 1. Clean markdown
    cleaned = clean_markdown_for_speech(raw_text)

    # 2. Hausa orthography
    if lang.lower() == "ha":
        cleaned = normalize_hausa_orthography(cleaned)

    # 3. Expand numbers and symbols
    expanded = expand_numbers_and_symbols(cleaned, lang=lang)

    # 4. Sentence chunking
    sentences = split_into_sentences(expanded)
    return sentences
