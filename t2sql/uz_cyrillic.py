"""Uzbek Latin to Uzbek Cyrillic, for questions.

Values have to survive unchanged, because the SQL matches them in the database: quoted text, codes, and capitalised
names that the English question also has stay in Latin letters. So do English words the translator kept.

People often type Uzbek Cyrillic on a Russian keyboard, which has no ў қ ғ ҳ. russian_keyboard() makes that variant.
"""
import re

_APOSTROPHES = "'‘’ʻʼ`"
_GROUPS = {"yo'": "йў", "s'h": "сҳ", "o'": "ў", "g'": "ғ", "sh": "ш", "ch": "ч", "yo": "ё", "yu": "ю", "ya": "я",
           "ye": "е"}
_LETTERS = dict(zip("abdfghijklmnopqrstuvxyzcw", "абдфгҳижклмнопқрстувхйзсв"))
_VOWELS = "aeiou"
_TOKEN = re.compile("yo[{0}]|s[{0}]h|[og][{0}]|sh|ch|yo|yu|ya|ye|ts|.".format(_APOSTROPHES), re.IGNORECASE | re.DOTALL)
_QUOTED = re.compile(r"""(?<!\w)['‘’"“«]([^'‘’"”»]+)['‘’"”»](?!\w)""")
_WORD = re.compile(r"[^\W_]+(?:[-_][^\W_]+)*")
_FOREIGN = re.compile(r"w|c(?!h)", re.IGNORECASE)  # Uzbek Latin has no w or lone c: an English word the translator kept
_KEYBOARD = str.maketrans("ўқғҳЎҚҒҲ", "укгхУКГХ")


def _convert(text: str) -> str:
    # -ksiya and -nsiya are spelled -кция and -нция; versiya and missiya keep their с
    text = re.sub(r"(?<=[nk])siya", "tsiya", text, flags=re.IGNORECASE)
    text = re.sub(r"(sen|ok)tabr", r"\1tyabr", text, flags=re.IGNORECASE)
    out, prev = [], ""
    for m in _TOKEN.finditer(text):
        token, low = m.group(), m.group().lower().translate({ord(a): "'" for a in _APOSTROPHES})
        word_start = not prev.isalpha()
        if low in _GROUPS:
            mapped = _GROUPS[low]
        elif low == "ts":
            mapped = "ц" if word_start or text[m.end():m.end() + 3].lower() == "iya" else "тс"
        elif low == "e":
            mapped = "э" if word_start or prev.lower() in _VOWELS else "е"
        elif low in _LETTERS:
            mapped = _LETTERS[low]
        elif token in _APOSTROPHES and prev.isalpha() and text[m.end():m.end() + 1].isalpha():
            mapped = "ъ"
        else:
            mapped = token
        out.append(mapped.capitalize() if token[0].isupper() else mapped)
        prev = token[-1]
    return "".join(out)


def to_uz_cyrillic(latin: str, english: str = "") -> str:
    """english is the source question: its quoted values and capitalised words or codes stay as they are."""
    keep = {w for w in _WORD.findall(english) if not w.islower()}
    spans = [m.span() for m in _QUOTED.finditer(latin)]
    for m in _WORD.finditer(latin):
        word = m.group()
        if word in keep or re.fullmatch(r"[A-Z]{2,}\d*", word) or _FOREIGN.search(word):
            spans.append(m.span())
    # inner words of a quoted value are already covered, so keep only the outermost spans
    out, at = [], 0
    for start, end in sorted(spans):
        if start >= at:
            out += [_convert(latin[at:start]), latin[start:end]]
            at = end
    out.append(_convert(latin[at:]))
    return "".join(out)


def russian_keyboard(cyrillic: str) -> str:
    return cyrillic.translate(_KEYBOARD)
