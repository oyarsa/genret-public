"""Constants, thresholds, and compiled regex patterns for facet evaluation."""

import re

FACET_NAMES = ("problems", "methods", "results", "contributions")
"""Canonical facet category names in extraction order."""

SCORE_WEIGHTS = {
    "structure": 0.15,
    "grounding": 0.45,
    "redundancy": 0.20,
    "specificity": 0.20,
}
"""Weights for combining per-dimension scores into an overall quality score."""

MIN_ITEMS_PER_FACET = 1
"""Minimum acceptable items per facet category."""
MAX_ITEMS_PER_FACET = 5

"""Maximum acceptable items per facet category."""
MIN_ITEM_CHARS = 10
"""Minimum character length for a facet item to be considered valid."""
MAX_ITEM_CHARS = 500
"""Maximum character length for a facet item before it's flagged."""

MAX_SENTENCES_PER_ITEM = 3
"""Maximum sentences per facet item before it's flagged as too long."""

LOW_GROUNDING_THRESHOLD = 0.3
"""Remapped grounding score below which an item is flagged as ungrounded."""
GROUNDING_FLOOR = 0.50
"""Raw cosine similarity mapped to 0. OpenAI embeddings for related text land in 0.4-0.9."""
GROUNDING_CEILING = 0.80
"""Raw cosine similarity mapped to 1."""
INTRA_DUP_THRESHOLD = 0.92
"""Cosine similarity above which two items within a facet are flagged as duplicates."""
CROSS_OVERLAP_THRESHOLD = 0.90
"""Cosine similarity above which items across facets are flagged as overlapping."""

EVIDENCE_CLIP_CHARS = 200
"""Maximum characters for evidence snippets in output."""

EVIDENCE_FIDELITY_THRESHOLD = 0.80
"""Coverage ratio at or above which an evidence string is considered faithful."""

EVIDENCE_MIN_MATCH_BLOCK = 8
"""Default minimum character length for a SequenceMatcher block to count.

Blocks shorter than this are treated as coincidental character overlap (e.g. common
short words, punctuation, whitespace) and excluded from the coverage sum.  This prevents
scattered tiny matches from inflating the score while still crediting multiple
substantial matching regions — for instance, a 100-char evidence string with one word
swapped in the middle produces two large blocks that both count.
"""

META_DESCRIPTION_PATTERNS = [
    r"^(?:the |this )(?:paper|work|study|research|article)",
    r"^(?:the )?authors?(?:\s)",
    r"^it (?:investigates?|explores?|examines?|addresses|highlights?|provides?|introduces?|employs?|utilizes?|proposes?|presents?|questions?|identifies?|also|further)",
    r"^(?:the |this )(?:proposed|presented|introduced)",
    r"^(?:the )?(?:study|analysis|evaluation|experiments?|findings?) (?:show|demonstrate|indicate|suggest|reveal|provide|highlight|introduce)",
]
"""Regexes matching meta-descriptions like 'The paper proposes X' instead of just 'X'."""

_COMPILED_META = [re.compile(p, re.IGNORECASE) for p in META_DESCRIPTION_PATTERNS]
"""Pre-compiled meta-description regexes."""

GENERIC_PATTERNS = [
    r"(?:addresses|investigates|explores|examines|presents) (?:the |a )?(?:challenge|problem|issue|gap|limitation|question|task)",
    r"the proposed (?:method|approach|model|framework|system|technique)",
    r"(?:significantly|substantially|greatly) (?:improves?|outperforms?|enhances?)",
    r"state[- ]of[- ]the[- ]art",
    r"the (?:main|key|primary|major) (?:contribution|finding|insight|takeaway)",
    r"novel (?:approach|method|framework|technique|model)",
    r"comprehensive (?:evaluation|analysis|experiments?|study)",
    r"the results (?:show|demonstrate|indicate|suggest|reveal)",
    r"outperform(?:s|ing)? (?:existing|previous|prior|baseline|other)",
    r"extensive experiments",
    r"achieves? (?:competitive|comparable|superior|strong|good|promising|better) (?:results?|performance)",
    r"(?:enhances?|advances?) the (?:field|understanding|state|knowledge)",
    r"contribut(?:e[sd]?|ing|ion) to the (?:field|understanding|literature|body)",
    r"(?:shedding|sheds?) light on",
    r"paving the way for future",
]
"""Regexes matching vague academic boilerplate that indicates low specificity."""

_COMPILED_GENERIC = [re.compile(p, re.IGNORECASE) for p in GENERIC_PATTERNS]
"""Pre-compiled generic-phrase regexes."""

FACET_TYPE_KEYWORDS = {
    "problems": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"challenge|difficulty|limitation|shortcoming",
            r"problem|issue|gap|lack",
            r"how (?:to|can|do)",
            r"(?:research )?question",
            r"remain(?:s|ing) (?:un(?:solved|addressed|clear))",
            r"(?:insufficien|inadequa)t",
        ]
    ],
    "methods": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"propose[sd]?|introduce[sd]?|develop[sed]*|design[sed]*",
            r"(?:training|learning|decoding) (?:strategy|procedure|algorithm)",
            r"architecture|framework|pipeline|module",
            r"fine[- ]tun|pre[- ]train",
            r"(?:we |the authors? )(?:use|employ|adopt|leverage|utilize)",
            r"encoder|decoder|transformer|attention",
        ]
    ],
    "results": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\d+\.?\d*\s*%",
            r"(?:f1|accuracy|precision|recall|bleu|rouge)\s*(?:score)?",
            r"outperform|surpass|exceed|improve[sd]? (?:over|upon|by)",
            r"(?:achieve|obtain|reach)[sd]?\s+(?:an? )?\w*\s*(?:of\s+)?\d",
            r"(?:best|highest|lowest|superior) (?:result|performance|score)",
            r"(?:baseline|benchmark|prior work|previous)",
        ]
    ],
    "contributions": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"contribut(?:e|ion|ing)",
            r"(?:first|novel|new|original) (?:to our knowledge)?",
            r"we (?:are the first|release|open[- ]source|provide)",
            r"(?:main|key|primary) (?:insight|finding|takeaway)",
            r"advance(?:s|ment)",
            r"bridge[sd]? (?:the )?gap",
        ]
    ],
}
"""Per-facet keyword patterns used to predict which category a facet item belongs to."""
