import os
from unidecode import unidecode
from nltk.corpus import wordnet as wn

TRANSCRIPTION_DATASET_PATH = "data/sentiment-analysis/manual-transcriptions"

def load_transcriptions():
    for filename in os.listdir(TRANSCRIPTION_DATASET_PATH):
        with open(os.path.join(TRANSCRIPTION_DATASET_PATH, filename), "r") as f:
            yield filename, f.read()

def add_accents(word, mapping=None):
    """
    Attempt to restore accents to an unaccented Spanish word.
    """
    # Use custom mapping for ambiguous words
    if mapping and word in mapping:
        return mapping[word][0]  # Choose the first option (or handle via context)
    
    # Return the word as is (fallback) or with accents from WordNet
    synsets = wn.synsets(word, lang='spa')
    if synsets:
        lemma_names = synsets[0].lemma_names('spa')
        for lemma in lemma_names:
            if lemma != unidecode(lemma):  # Check if lemma has accents
                return lemma
    return word 