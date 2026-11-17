"""Stopword list for facet n-gram Jaccard similarity computation.

Terms in this set are excluded when building n-gram features from facet items. Derived
from IDF analysis over the splits-connected-10k corpus (10,000 papers), manually
reviewed to remove discriminative terms.
"""

# fmt: off
STOPWORDS = frozenset({
    # Function words
    "a", "across", "all", "an", "and", "are", "as", "at", "be", "between", "both",
    "but", "by", "each", "for", "from", "how", "in", "including", "into", "is",
    "it", "many", "more", "most", "no", "not", "of", "often", "on", "only", "or",
    "over", "per", "so", "some", "than", "that", "the", "their", "there", "to",
    "via", "when", "where", "whether", "which", "while", "with", "without",
    # Tokenisation artefacts (from e.g., ...)
    "e", "g",
    # Number tokens
    "0", "1", "2", "3", "5", "one", "two", "three", "four", "five",
    # Generic verbs / actions
    "design", "do", "evaluate", "improve", "improves", "introduce", "produce",
    "provide", "show", "showing", "train", "use", "using",
    # Generic result / comparison language
    "best", "better", "different", "effective", "existing", "higher", "large",
    "multiple", "new", "novel", "plus", "prior", "simple", "small", "standard",
    "strong", "vs",
    # Generic nouns (too broad to discriminate)
    "approaches", "baseline", "data", "dataset", "examples", "gains", "information",
    "method", "methods", "outputs", "pairs", "pipeline", "quality", "representations",
    "results", "scores", "sentence", "set", "settings", "tasks", "text",
    # Generic descriptors / discourse markers
    "can", "compared", "demonstrating", "especially", "indicating", "lack",
    "learning", "outperforms", "state-of-the-art", "substantially", "test",
    "yields",
})
# fmt: on
