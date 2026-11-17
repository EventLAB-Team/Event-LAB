"""EventLAB terminal palette: pure black, hairline rules, magenta and cyan accents.

Values follow the dark ink scale used by omp.sh (converted from OKLCH), nudged
where terminal rendering needs a little more contrast than a browser page.
"""
# Ink scale
BLACK = '#000000'
ELEVATED = '#0a0a0c'      # panels, dialogs, footer
RAISED = '#141418'        # inactive cursor rows, hover
RULE = '#202024'          # hairline borders
RULE_STRONG = '#34343a'   # dialog borders, scrollbar thumbs
FAINT = '#626368'         # numbers, eyebrows, key hints
MUTED = '#8e8f95'         # secondary copy
DIM = '#c3c4c7'           # body copy
FG = '#f5f5f5'            # headings, values

# Accents
MAGENTA = '#f84fcc'
MAGENTA_DEEP = '#2a0e24'  # focused cursor rows
CYAN = '#00dbe4'
CYAN_DIM = '#0d6a70'

# Home backdrop event polarities
EVENT_ON = '#4a86ff'
EVENT_OFF = '#ff4a5e'

# Availability states
COMPLETE = '#6ee7b7'
PARTIAL = '#fbbf72'
MISSING = '#ff7a8a'


def eventlab_theme():
    from textual.theme import Theme
    return Theme(
        name='eventlab', primary=MAGENTA, secondary=CYAN, accent=MAGENTA,
        foreground=FG, background=BLACK, surface=BLACK, panel=ELEVATED, boost=RAISED,
        success=COMPLETE, warning=PARTIAL, error=MISSING, dark=True,
        variables={
            'rule': RULE, 'rule-strong': RULE_STRONG, 'raised': RAISED, 'elevated': ELEVATED,
            'faint': FAINT, 'muted': MUTED, 'dim': DIM, 'magenta-deep': MAGENTA_DEEP,
            'cyan': CYAN, 'cyan-dim': CYAN_DIM,
            'block-cursor-text-style': 'bold',
            'footer-background': BLACK, 'footer-item-background': BLACK,
            'footer-key-foreground': MAGENTA, 'footer-key-background': 'transparent',
            'footer-description-foreground': MUTED, 'footer-description-background': 'transparent',
            'scrollbar': RULE_STRONG, 'scrollbar-hover': FAINT, 'scrollbar-active': MAGENTA,
            'scrollbar-background': BLACK, 'scrollbar-background-hover': BLACK,
            'scrollbar-background-active': BLACK, 'scrollbar-corner-color': BLACK,
            'input-cursor-background': MAGENTA, 'input-selection-background': f'{MAGENTA} 30%',
            'link-color': CYAN, 'link-background': 'transparent', 'link-style': 'none',
            'link-color-hover': FG, 'link-background-hover': 'transparent', 'link-style-hover': 'underline',
        })


def eyebrow(text):
    """Letter-spaced uppercase label, the terminal take on 0.2em tracking."""
    return ' '.join(text.upper())
