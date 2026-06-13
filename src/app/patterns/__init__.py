"""Detection of classical geometric chart patterns.

Modules:
    interfaces — PatternDetector Protocol
    triangles  — TRIANGLE_ASC / DESC / SYM
    rectangles — RECTANGLE (horizontal range)
    channels   — CHANNEL_UP / DOWN
    wedges     — WEDGE_RISING / FALLING
    flags      — FLAG_BULL / BEAR (pole + consolidation)
    reversal   — DOUBLE_TOP / BOTTOM, HEAD_SHOULDERS, INVERSE_HEAD_SHOULDERS

All detectors respect the "no look-ahead" rule: at a candle index ``i``
only pivots confirmed at ``i`` (or before) are used.
"""
