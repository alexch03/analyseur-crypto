"""Détection de patterns chartistes géométriques classiques.

Modules :
    interfaces — Protocol PatternDetector
    triangles  — TRIANGLE_ASC / DESC / SYM
    rectangles — RECTANGLE (range horizontal)
    channels   — CHANNEL_UP / DOWN
    wedges     — WEDGE_RISING / FALLING
    flags      — FLAG_BULL / BEAR (pole + consolidation)
    reversal   — DOUBLE_TOP / BOTTOM, HEAD_SHOULDERS, INVERSE_HEAD_SHOULDERS

Tous les détecteurs respectent la règle "no look-ahead" : à un index de bougie ``i``
seuls les pivots confirmés à ``i`` (ou avant) sont utilisés.
"""
