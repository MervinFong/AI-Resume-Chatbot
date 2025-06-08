import spacy
from spacy.cli import download

try:
    spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
