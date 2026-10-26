"""Train the German part-of-speech tagger used by the pipeline.

Produces article_collector/nltk_german_classifier_data.pickle, which is
already included in the repository. To retrain it, download the TIGER corpus
(CoNLL-09 format) from the University of Stuttgart (license does not allow
redistribution) and run this script from the repository root:

    python scripts/train_pos_tagger.py path/to/tiger_release_aug07.corrected.16012013.conll09
"""

import pickle
import random
import sys
from pathlib import Path

import nltk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger  # noqa: E402

corpus_file = Path(sys.argv[1])
corp = nltk.corpus.ConllCorpusReader(
    str(corpus_file.parent), corpus_file.name,
    ["ignore", "words", "ignore", "ignore", "pos"],
    encoding="utf-8",
)

tagged_sents = list(corp.tagged_sents())
random.seed(42)
random.shuffle(tagged_sents)

# 90 % training, 10 % test
split_size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]

tagger = ClassifierBasedGermanTagger(train=train_sents)
print(f"Test accuracy: {tagger.accuracy(test_sents):.3f}")

out = Path(__file__).resolve().parents[1] / "article_collector" / "nltk_german_classifier_data.pickle"
with open(out, "wb") as f:
    pickle.dump(tagger, f, protocol=2)
