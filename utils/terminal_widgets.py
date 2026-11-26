"""Compact terminal controls with explicit keyboard and selection states."""
import math
import os
import random

from rich.text import Text
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip
from textual import on
from textual.binding import Binding
from textual.message import Message
from textual.widgets import OptionList, DataTable, Input, Static, Select, Link
from textual.widgets.option_list import Option
from textual.screen import ModalScreen
from textual.containers import Vertical, VerticalScroll
from textual.widget import Widget

from utils.terminal_theme import (BLACK, CYAN, DIM, EVENT_OFF, EVENT_ON, FAINT, FG, MAGENTA, MUTED, RULE,
                                  eyebrow)


class EventLogo(Static):
    """Block wordmark. A diagonal light sweeps across it every few seconds."""
    LOGO = (
        '█████ █   █ █████ █   █ █████  █      ███  ████ ',
        '█     █   █ █     ██  █   █    █     █   █ █   █',
        '████  █   █ ████  █ █ █   █    █     █████ ████ ',
        '█      █ █  █     █  ██   █    █     █   █ █   █',
        '█████   █   █████ █   █   █    █████ █   █ ████ ',
    )
    SPLIT = 29  # EVENT | LAB

    # Five-row lettering with two columns of space between subtitle letters.
    LETTERS = {
        'L': ('█  ', '█  ', '█  ', '█  ', '███'),
        'O': ('███', '█ █', '█ █', '█ █', '███'),
        'C': ('███', '█  ', '█  ', '█  ', '███'),
        'A': ('███', '█ █', '███', '█ █', '█ █'),
        'I': ('███', ' █ ', ' █ ', ' █ ', '███'),
        'Z': ('███', '  █', ' █ ', '█  ', '███'),
        'T': ('███', ' █ ', ' █ ', ' █ ', ' █ '),
        'N': ('█  █', '██ █', '█ ██', '█  █', '█  █'),
    }
    TICK = .08     # seconds per phase step
    PERIOD = 110   # steps per sweep cycle (~9 s, as on omp.sh)
    SWEEP = 26     # steps the light takes to cross

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = self.PERIOD - 8
        subtitle = ['  '.join(self.LETTERS[letter][row] for letter in 'LOCALIZATION') for row in range(5)]
        word = len(subtitle[0])
        subtitle[4] += '  █'  # a magenta full stop, like the omp.sh headline
        self.width = max(len(self.LOGO[0]), len(subtitle[4]))
        # Each canvas row: (text, [(start, end, colour)]).
        self.rows = [(line, [(0, self.SPLIT, FG), (self.SPLIT, len(line), MAGENTA)]) for line in self.LOGO]
        self.rows.append(('', []))
        self.rows += [(line, [(0, word, DIM), (word, len(line), MAGENTA)]) for line in subtitle]

    def on_mount(self):
        self.draw_logo()
        self.animation = self.set_interval(self.TICK, self.advance, pause=True)
        self.animation.resume()

    def advance(self):
        self.phase = (self.phase + 1) % self.PERIOD
        if self.phase <= self.SWEEP + 1:
            self.draw_logo()

    def light(self, row, column):
        """0..1 brightness of the sweep at a cell; the band leans like a 105° gradient."""
        if self.phase > self.SWEEP:
            return 0
        centre = (self.phase / self.SWEEP) * (self.width + 24) - 12
        distance = abs(column + row * .6 - centre)
        return max(0., 1 - distance / 5)

    def draw_logo(self):
        logo = Text(no_wrap=True, overflow='crop')
        for row, (line, spans) in enumerate(self.rows):
            for start, end, colour in spans:
                for column in range(start, end):
                    glyph = line[column]
                    glow = self.light(row, column) if glyph != ' ' else 0
                    style = (CYAN if glow > .55 else blend(colour, CYAN, glow * 1.4)) if glow else colour
                    logo.append(glyph, style)
            logo.append(' ' * (self.width - len(line)) + ('\n' if row < len(self.rows) - 1 else ''))
        self.update(logo)


class HomeMenu(OptionList):
    BINDINGS = [Binding('down', 'cursor_down', 'Choose', key_display='↑/↓'),
                Binding('enter', 'select', 'Open')]

    @staticmethod
    def item(number, title, description):
        return Text.assemble((f'{number:02d}  ', FAINT), (title, 'bold'), '\n',
                             ('    ' + description, MUTED), no_wrap=True, overflow='ellipsis')

    def on_mount(self):
        self.border_title = eyebrow('Start')
        self.border_subtitle = '↑↓ choose · ↵ open'


class SectionTabs(Static):
    """Uppercase section tabs over a hairline; the active one is underlined in magenta."""
    TABS = (('configuration', 'Configure'), ('experiments', 'Experiments'), ('downloads', 'Downloads'),
            ('logs', 'Activity'), ('results', 'Results'), ('live', 'Live'))
    GAP = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active = None
        self.spans = []

    def show(self, active):
        self.active = active
        self.refresh()

    def render(self):
        width = max(self.size.width, 1)
        labels, rule, x = Text(no_wrap=True, overflow='crop'), Text(no_wrap=True), 1
        labels.append(' ')
        self.spans = []
        for key, title in self.TABS:
            label = title.upper()
            labels.append(label, f'bold {FG}' if key == self.active else MUTED)
            labels.append(' ' * self.GAP)
            self.spans.append((x - 1, x + len(label) + 1, key))
            x += len(label) + self.GAP
        hint = 'ctrl ←→'
        if width - x > len(hint) + 2:
            labels.append(' ' * (width - x - len(hint) - 1) + hint, FAINT)
        cursor = 0
        for start, end, key in self.spans:
            if key == self.active:
                rule.append('─' * start, RULE)
                rule.append('━' * (end - start), MAGENTA)
                cursor = end
        rule.append('─' * max(0, width - cursor), RULE)
        return Text('\n').join([labels, rule])

    def on_click(self, event):
        for start, end, key in self.spans:
            if start <= event.x < end:
                self.app.query_one('#tabs').active = key


class Panel(Vertical):
    """Hairline column with an eyebrow-style border title."""
    def __init__(self, number, title, *children, **kwargs):
        super().__init__(*children, **kwargs)
        self.border_title = f'[{FAINT}]{number:02d}[/]  {eyebrow(title)}'


class SelectionList(OptionList):
    BINDINGS = [('space', 'toggle', 'Select'),
                ('a', 'toggle_all', 'All/none'),
                Binding('left', 'app.pane(-1)', 'Previous panel', show=False),
                Binding('right', 'app.pane(1)', 'Panel', key_display='←/→')]

    class SelectedChanged(Message):
        def __init__(self, listing):
            super().__init__()
            self.selection_list = listing

    def __init__(self, *items, **kwargs):
        self.items = list(items)
        self.selected = [item[1] for item in items if len(item) > 2 and item[2]]
        super().__init__(*[Option(self.prompt(item)) for item in items], **kwargs)

    def prompt(self, item):
        label = item[0] if isinstance(item[0], Text) else Text(str(item[0]))
        chosen = item[1] in self.selected
        prompt = Text(no_wrap=True, overflow='ellipsis')
        prompt.append('■ ' if chosen else '□ ', MAGENTA if chosen else FAINT)
        prompt.append_text(label)
        return prompt

    def select(self, value):
        if value not in self.selected:
            self.selected.append(value)
            self.changed()

    def deselect_all(self):
        self.selected.clear()
        self.changed()

    def action_toggle_all(self):
        self.selected = [] if len(self.selected) == len(self.items) else [item[1] for item in self.items]
        self.changed()

    def toggle(self, value):
        if value in self.selected:
            self.selected.remove(value)
        else:
            self.selected.append(value)
        self.changed()

    def changed(self):
        for index, item in enumerate(self.items):
            self.replace_option_prompt_at_index(index, self.prompt(item))
        self.post_message(self.SelectedChanged(self))

    def action_toggle(self):
        if self.highlighted is not None:
            self.toggle(self.items[self.highlighted][1])

    @on(OptionList.OptionSelected)
    def picked(self, event):
        event.stop()
        self.action_toggle()


class CompactTable(DataTable):
    BINDINGS = [Binding('left', 'pane(-1)', 'Previous panel', show=False),
                Binding('right', 'pane(1)', 'Panel', key_display='←/→')]

    def action_pane(self, direction):
        self.app.action_pane(direction)


class RoleTable(CompactTable):
    BINDINGS = [('r', 'role("R")', 'Reference'), ('q', 'role("Q")', 'Query'),
                ('delete', 'role("")', 'Exclude'), ('space', 'role("cycle")', 'Cycle role'),
                ('a', 'app.toggle_all_roles', 'All/none')]

    def action_role(self, role):
        self.app.set_role(role)


class ValueEditor(ModalScreen):
    BINDINGS = [('escape', 'dismiss(None)', 'Cancel')]
    def __init__(self, key, value, choices=None, description='', disabled_choices=(), docs_url=None):
        super().__init__()
        self.key, self.value, self.choices = key, value, choices
        self.description = description
        self.disabled_choices = set(disabled_choices)
        self.docs_url = docs_url

    def compose(self):
        with Vertical(id='value-dialog') as dialog:
            dialog.border_title = eyebrow(self.key) if len(self.key) < 24 else self.key.upper()
            if self.description:
                yield Static(self.description, markup=False, classes='muted')
            if self.docs_url:
                yield Link('reader.with_repr() ↗', url=self.docs_url, id='options-docs')
            if self.choices:
                yield Select([(Text(f'{v} — unavailable', style='dim') if v in self.disabled_choices else v, v)
                              for v in self.choices], value=self.value,
                             allow_blank=False, id='value-input', compact=True)
            else:
                yield Input(self.value, id='value-input', compact=True)
            yield Static('↑/↓ choose · Enter select · Esc cancel' if self.choices
                         else 'Enter accept · Esc cancel', classes='muted')

    def on_mount(self):
        widget = self.query_one('#value-input')
        widget.focus()
        if self.choices:
            options = widget.query_one(OptionList)
            for index, value in enumerate(self.choices):
                if value in self.disabled_choices:
                    options.disable_option_at_index(index)
            widget.action_show_overlay()

    @on(Input.Submitted)
    def accept(self, event):
        self.dismiss(event.value)

    @on(Select.Changed)
    def choose(self, event):
        if event.value != self.value and event.value not in self.disabled_choices:
            self.dismiss(str(event.value))


def blend(a, b, t):
    """Mix two #rrggbb colours."""
    t = max(0., min(1., t))
    mixed = (round(int(a[i:i + 2], 16) * (1 - t) + int(b[i:i + 2], 16) * t) for i in (1, 3, 5))
    return '#' + ''.join(f'{v:02x}' for v in mixed)


def noise(x, y, salt=0):
    """Stable 0..1 hash so the backdrop is identical on every redraw."""
    n = (x * 374761393 + y * 668265263 + salt * 2246822519) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return ((n ^ (n >> 16)) & 0xffffff) / 0x1000000


class EventField(VerticalScroll):
    """Home backdrop: a sparse, simulated event-camera stream.

    A few unseen objects drift across the view. Like a real event sensor, only
    their moving edges fire: the leading edge emits ON events (blue), the
    trailing edge OFF events (red), and edges parallel to the motion stay
    silent. Nothing fires right beside the content column. A trickle of background-activity noise fires everywhere. Events are
    half-block pixels that fade over about a second. Rows without events reuse a
    cached grain strip, so each frame only rebuilds the handful of lit rows.
    """
    TICK = .1
    LIFE = 9                 # frames an event stays visible
    NOISE = .00035           # background events per pixel per frame
    EDGE_RATE = .07          # events per pixel of moving contour per frame

    def __init__(self, *children, **kwargs):
        super().__init__(*children, **kwargs)
        self.random = random.Random(7)
        self.events = {}     # (x, y) pixel -> (age, polarity)
        self.objects = []
        self._style_cache = {}
        self.grain_key, self.grain = None, []
        self.quiet = (0, 0)  # columns kept free of events around the content
        # Fade ramps in the usual DVS polarity colours (blue ON, red OFF), kept
        # dim and away from the interface's magenta/cyan so text always wins.
        self.ramps = {pol: [blend(BLACK, colour, .5 * (1 - age / self.LIFE) ** 1.5 + .05)
                            for age in range(self.LIFE)]
                      for pol, colour in ((1, EVENT_ON), (-1, EVENT_OFF))}

    def on_mount(self):
        self.animation = self.set_interval(self.TICK, self.advance)

    def spawn(self, width, height, anywhere=False):
        """An object of radius r moving slowly; it enters from a side unless anywhere."""
        r = self.random.uniform(4, 11)
        speed = self.random.uniform(.6, 1.4)
        angle = self.random.uniform(-.5, .5) + (0 if self.random.random() < .5 else 3.1416)
        vx, vy = speed * math.cos(angle), speed * math.sin(angle) * .6
        x = self.random.uniform(0, width) if anywhere else (-r if vx > 0 else width + r)
        y = self.random.uniform(r, max(r + 1, height - r))
        return [x, y, vx, vy, r]

    def advance(self):
        width, height = self.size.width, self.size.height * 2
        if width < 2 or height < 2:
            return
        if self.children and self.children[0].region.width:
            content = self.children[0].region
            self.quiet = (content.x - self.region.x - 4, content.right - self.region.x + 4)
        aged = {}
        for key, (age, pol) in self.events.items():
            if age + 1 < self.LIFE:
                aged[key] = (age + 1, pol)
        self.events = aged
        while len(self.objects) < max(2, width * height // 3200):
            self.objects.append(self.spawn(width, height, anywhere=not self.objects))
        alive = []
        for body in self.objects:
            x, y, vx, vy, r = body
            speed = math.hypot(vx, vy)
            for _ in range(int(2 * math.pi * r * self.EDGE_RATE + self.random.random())):
                theta = self.random.uniform(0, 2 * math.pi)
                nx, ny = math.cos(theta), math.sin(theta)
                flow = (nx * vx + ny * vy) / speed
                if abs(flow) < .3:
                    continue  # edge parallel to motion: no brightness change
                self.fire(int(x + nx * r), int(y + ny * r), 1 if flow > 0 else -1, width, height)
            body[0], body[1] = x + vx, y + vy
            if -2 * r < body[0] < width + 2 * r and -2 * r < body[1] < height + 2 * r:
                alive.append(body)
        self.objects = alive
        for _ in range(int(width * height * self.NOISE + self.random.random())):
            self.fire(self.random.randrange(width), self.random.randrange(height),
                      self.random.choice((1, -1)), width, height)
        self.refresh()

    def fire(self, x, y, polarity, width, height):
        if 0 <= x < width and 0 <= y < height and not self.quiet[0] <= x < self.quiet[1]:
            self.events[(x, y)] = (0, polarity)

    def colour(self, x, y):
        event = self.events.get((x, y))
        return self.ramps[event[1]][event[0]] if event else None

    def style(self, fg, bg):
        key = (fg, bg)
        if key not in self._style_cache:
            self._style_cache[key] = Style(color=fg, bgcolor=bg or BLACK)
        return self._style_cache[key]

    def grain_row(self, row, width):
        if self.grain_key != width:
            self.grain_key = width
            self.grain = {}
        if row not in self.grain:
            text = ''.join('·' if noise(x, row, 9) < 1 / 40 else ' ' for x in range(width))
            self.grain[row] = text
        return self.grain[row]

    def render_line(self, y):
        width = self.size.width
        grain = self.grain_row(y, width)
        lit = sorted({x for (x, py) in self.events if py in (2 * y, 2 * y + 1)})
        if not lit:
            return Strip([Segment(grain, self.style('#1a1a1e', None))], width)
        segments, cursor = [], 0
        for x in lit:
            if x > cursor:
                segments.append(Segment(grain[cursor:x], self.style('#1a1a1e', None)))
            top, bottom = self.colour(x, 2 * y), self.colour(x, 2 * y + 1)
            if top and bottom:
                segments.append(Segment('▀', self.style(top, bottom)))
            elif top:
                segments.append(Segment('▀', self.style(top, None)))
            else:
                segments.append(Segment('▄', self.style(bottom, None)))
            cursor = x + 1
        if cursor < width:
            segments.append(Segment(grain[cursor:], self.style('#1a1a1e', None)))
        return Strip(segments, width)


class PixelView(Widget):
    """An RGB image, fitted to the widget with its aspect kept and centred.

    In Kitty it is real pixels, as `eventcv tui` draws them: each `show` uploads the image once,
    deflated, at its own resolution, with a virtual placement the size of the view, and the view's
    cells are Kitty's Unicode placeholders, coloured with the image id. The terminal does the
    scaling, so a frame costs one small upload rather than a styled cell per pixel pair. Three ids
    take turns, so each frame replaces one the screen is no longer showing. Elsewhere (or inside
    tmux, whose passthrough this does not wrap) it falls back to half blocks.
    `show(image)` takes an (H, W, 3) uint8 array, or None to clear.
    """
    DEFAULT_CSS = 'PixelView { height: 1fr; width: 1fr; background: $background; }'
    _views = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image, self.fitted, self.fitted_for = None, None, None
        self._styles = {}
        self.kitty = KITTY_GRAPHICS
        base = 0x00ec0001 + 3 * PixelView._views
        PixelView._views += 1
        self.ids, self.turn, self.placed = [base, base + 1, base + 2], 0, None

    def show(self, image):
        self.image, self.fitted_for = image, None
        self.placed = None
        if self.kitty and image is not None and self.size.width and self.size.height:
            self.transmit()
        self.refresh()

    def on_resize(self):
        if self.image is not None:
            self.show(self.image)

    def on_unmount(self):
        if self.kitty:
            self.write(''.join(f'\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\' for image_id in self.ids))

    def write(self, sequence):
        try:
            self.app._driver.write(sequence)
            return True
        except Exception:  # no terminal (tests, headless): draw half blocks instead
            self.kitty = False
            return False

    def placement(self):
        """Cells (columns, rows) the image fills, and its offset, for cells twice as tall as wide."""
        width, height = self.size.width, self.size.height
        h, w = self.image.shape[:2]
        columns, rows = width, max(1, round(width * h / w / 2))
        if rows > height:
            rows, columns = height, max(1, round(height * 2 * w / h))
        return columns, min(rows, len(KITTY_DIACRITICS)), (width - columns) // 2, (height - rows) // 2

    def transmit(self):
        import base64
        import zlib

        import numpy as np

        columns, rows, left, top = self.placement()
        image_id = self.ids[self.turn % len(self.ids)]
        self.turn += 1
        h, w = self.image.shape[:2]
        # Fast deflate: the frames are mostly black, so it costs little and runs every frame.
        payload = base64.standard_b64encode(zlib.compress(np.ascontiguousarray(self.image, np.uint8).tobytes(), 1))
        chunks = [payload[i:i + 4096] for i in range(0, len(payload), 4096)] or [b'']
        sequence = []
        for index, chunk in enumerate(chunks):
            keys = (f'i={image_id},a=T,U=1,f=24,o=z,t=d,s={w},v={h},c={columns},r={rows},'
                    if index == 0 else '')
            sequence.append(f'\x1b_Gq=2,{keys}m={int(index + 1 < len(chunks))};{chunk.decode()}\x1b\\')
        if self.write(''.join(sequence)):
            self.placed = (image_id, columns, rows, left, top)

    def fit(self):
        import numpy as np

        key = (id(self.image), self.size.width, self.size.height)
        if self.fitted_for == key:
            return self.fitted
        width, height = self.size.width, self.size.height * 2
        h, w = self.image.shape[:2]
        scale = min(width / w, height / h)
        fw, fh = max(1, int(w * scale)), max(2, int(h * scale) // 2 * 2)
        rows = np.minimum((np.arange(fh) / scale).astype(int), h - 1)
        cols = np.minimum((np.arange(fw) / scale).astype(int), w - 1)
        small = (self.image[rows][:, cols] >> 3) << 3
        self.fitted = (small, (width - fw) // 2, (height - fh) // 4)
        self.fitted_for = key
        return self.fitted

    def style(self, top, bottom):
        key = (top, bottom)
        if key not in self._styles:
            self._styles[key] = Style(color='#%02x%02x%02x' % top, bgcolor='#%02x%02x%02x' % bottom)
        return self._styles[key]

    def render_line(self, y):
        width, background = self.size.width, self.rich_style
        if self.image is None:
            return Strip.blank(width, background)
        if self.placed:
            image_id, columns, rows, left, top = self.placed
            row = y - top
            if not 0 <= row < rows:
                return Strip.blank(width, background)
            # The first cell names its row (then column 0 and the id's top byte, 0 for these ids);
            # Kitty infers the columns after it.
            first = KITTY_PLACEHOLDER + chr(KITTY_DIACRITICS[row]) + chr(KITTY_DIACRITICS[0]) * 2
            colour = Style(color='#%06x' % (image_id & 0xffffff), bgcolor=background.bgcolor)
            return Strip([Segment(' ' * left, background),
                          Segment(first + KITTY_PLACEHOLDER * (columns - 1), colour)],
                         width).extend_cell_length(width, background)
        small, left, top_row = self.fit()
        row = y - top_row
        if not 0 <= row < small.shape[0] // 2:
            return Strip.blank(width, background)
        upper, lower = small[2 * row], small[2 * row + 1]
        segments, run, start = [], None, 0
        for x in range(small.shape[1] + 1):
            pair = (tuple(upper[x]), tuple(lower[x])) if x < small.shape[1] else None
            if pair != run:
                if run is not None:
                    segments.append(Segment('▀' * (x - start), self.style(*run)))
                run, start = pair, x
        return Strip([Segment(' ' * left, background)] + segments, width).extend_cell_length(width, background)


# Kitty's graphics protocol, as `eventcv tui` uses it. EVENTLAB_GRAPHICS=kitty|halfblocks overrides
# the detection.
KITTY_GRAPHICS = (os.environ.get('EVENTLAB_GRAPHICS') == 'kitty' or (
    os.environ.get('EVENTLAB_GRAPHICS') != 'halfblocks' and 'TMUX' not in os.environ
    and (os.environ.get('TERM') == 'xterm-kitty' or 'KITTY_WINDOW_ID' in os.environ)))
KITTY_PLACEHOLDER = '\U0010EEEE'
# The row/column diacritics, `rowcolumn-diacritics.txt` from Kitty's graphics protocol (the n-th
# marks row or column n), as eventcv's tui/pixels.rs copies them from ratatui-image (MIT).
KITTY_DIACRITICS = (
    0x0305, 0x030D, 0x030E, 0x0310, 0x0312, 0x033D, 0x033E, 0x033F, 0x0346, 0x034A, 0x034B, 0x034C,
    0x0350, 0x0351, 0x0352, 0x0357, 0x035B, 0x0363, 0x0364, 0x0365, 0x0366, 0x0367, 0x0368, 0x0369,
    0x036A, 0x036B, 0x036C, 0x036D, 0x036E, 0x036F, 0x0483, 0x0484, 0x0485, 0x0486, 0x0487, 0x0592,
    0x0593, 0x0594, 0x0595, 0x0597, 0x0598, 0x0599, 0x059C, 0x059D, 0x059E, 0x059F, 0x05A0, 0x05A1,
    0x05A8, 0x05A9, 0x05AB, 0x05AC, 0x05AF, 0x05C4, 0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615,
    0x0616, 0x0617, 0x0657, 0x0658, 0x0659, 0x065A, 0x065B, 0x065D, 0x065E, 0x06D6, 0x06D7, 0x06D8,
    0x06D9, 0x06DA, 0x06DB, 0x06DC, 0x06DF, 0x06E0, 0x06E1, 0x06E2, 0x06E4, 0x06E7, 0x06E8, 0x06EB,
    0x06EC, 0x0730, 0x0732, 0x0733, 0x0735, 0x0736, 0x073A, 0x073D, 0x073F, 0x0740, 0x0741, 0x0743,
    0x0745, 0x0747, 0x0749, 0x074A, 0x07EB, 0x07EC, 0x07ED, 0x07EE, 0x07EF, 0x07F0, 0x07F1, 0x07F3,
    0x0816, 0x0817, 0x0818, 0x0819, 0x081B, 0x081C, 0x081D, 0x081E, 0x081F, 0x0820, 0x0821, 0x0822,
    0x0823, 0x0825, 0x0826, 0x0827, 0x0829, 0x082A, 0x082B, 0x082C, 0x082D, 0x0951, 0x0953, 0x0954,
    0x0F82, 0x0F83, 0x0F86, 0x0F87, 0x135D, 0x135E, 0x135F, 0x17DD, 0x193A, 0x1A17, 0x1A75, 0x1A76,
)
