"""Class names of the CoPeD zero-shot semantic labels, per sequence.

The released labels carry category ids only; each sequence has its own vocabulary. The names below
are a visual reading of the masks and are used for reports only; all code works with the ids.
"""

CLASS_NAMES = {
    "NYUARPL": ("ceiling", "floor", "wall", "chair", "door", "window", "light", "person", "table/furniture"),
}
