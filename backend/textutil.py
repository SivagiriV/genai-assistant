"""Small text helpers shared by the customer-facing apps."""
import re

# [2]  [1, 2]  【2】  【2†source】  (the model sometimes uses full-width brackets)
_CITATION = re.compile(r"[ \t]*[\[【]\s*\d+(?:\s*[,，、]\s*\d+)*\s*(?:†[^\]】]*)?[\]】]")


def strip_citations(text):
    return _CITATION.sub("", text)
