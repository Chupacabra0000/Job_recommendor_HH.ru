import re
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

# Very lightweight multilingual-friendly tokenizer.
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_+#\.\-]{2,}")

# Minimal stopwords (Russian + English). Extend if needed.
_STOPWORDS = {
    # RU
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты",
    "к","у","же","вы","за","бы","по","ее","мне","было","вот","от","меня","еще","нет","о","из","ему","теперь",
    "когда","даже","ну","вдруг","ли","если","уже","или","ни","быть","был","него","до","вас","нибудь","опять",
    "уж","вам","ведь","там","потом","себя","ничего","ей","может","они","тут","где","есть","надо","ней","для",
    "мы","тебя","их","чем","была","сам","чтоб","без","будто","чего","раз","тоже","себе","под","будет","ж",
    "тогда","кто","этот","того","потому","этого","какой","совсем","ним","здесь","этом","один","почти","мой",
    "тем","чтобы","нее","сейчас","были","куда","зачем","всех","никогда","можно","при","наконец","два","об",
    "другой","хоть","после","над","больше","тот","через","эти","нас","про","всего","них","какая","много","разве",
    "три","эту","моя","впрочем","хорошо","свою","этой","перед","иногда","лучше","чуть","том","нельзя","такой",
    "им","более","всегда","конечно","всю","между",
    # EN
    "the","and","for","with","from","that","this","are","was","were","have","has","had","you","your","our","they",
    "them","their","not","but","can","will","would","should","could","a","an","to","in","on","at","by","as","of",
    "or","if","is","it","we","i","me","my","he","she","his","her","be","been","being"
}

def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    toks = _TOKEN_RE.findall(text)
    toks = [t for t in toks if t not in _STOPWORDS and len(t) >= 2]
    return toks

def extract_terms(text: str, top_k: int = 10) -> List[str]:
    """Return top_k TF-IDF terms from a single document (resume text)."""
    text = (text or "").strip()
    if not text:
        return []
    tokens = _tokenize(text)
    if not tokens:
        return []
    # Build vectorizer on the single doc; use token pattern disabled since we pass pretokenized.
    vec = TfidfVectorizer(analyzer=lambda x: x, lowercase=False)
    X = vec.fit_transform([tokens])
    scores = X.toarray().ravel()
    terms = vec.get_feature_names_out()
    idx = scores.argsort()[::-1]
    out = []
    for i in idx:
        t = str(terms[i]).strip()
        if t and t not in out:
            out.append(t)
        if len(out) >= int(top_k):
            break
    return out
