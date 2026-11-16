"""Compact terminal controls with explicit keyboard and selection states."""
from rich.text import Text
from textual import on
from textual.binding import Binding
from textual.message import Message
from textual.widgets import OptionList, DataTable, Input, Static, Select, Link
from textual.widgets.option_list import Option
from textual.screen import ModalScreen
from textual.containers import Vertical


class EventLogo(Static):
    """A fixed-size wordmark; only the surrounding event sparks animate."""
    LOGO = (
        '█████ █   █ █████ █   █ █████  █      ███  ████ ',
        '█     █   █ █     ██  █   █    █     █   █ █   █',
        '████  █   █ ████  █ █ █   █    █     █████ ████ ',
        '█      █ █  █     █  ██   █    █     █   █ █   █',
        '█████   █   █████ █   █   █    █████ █   █ ████ ',
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = 0

    def on_mount(self):
        self.draw_logo()
        self.animation = self.set_interval(.35, self.advance, pause=True)
        self.animation.resume()

    def advance(self):
        self.phase = (self.phase + 1) % 12
        self.draw_logo()

    def draw_logo(self):
        logo = Text(justify='center', no_wrap=True)
        for row, line in enumerate(self.LOGO):
            left = ('·', '+', '✦', '+', '·', ' ')[(self.phase + row * 2) % 6]
            right = (' ', '·', '+', '✧', '+', '·')[(self.phase + row * 3) % 6]
            logo.append(f'{left}   ', '#56dce8')
            logo.append(line[:29], 'bold #ff718f')
            logo.append(line[29:], 'bold #56dce8')
            logo.append(f'   {right}' + ('\n' if row < 4 else ''), '#ff718f')
        self.update(logo)


class HomeMenu(OptionList):
    BINDINGS = [Binding('down', 'cursor_down', 'Choose', key_display='↑/↓'),
                Binding('enter', 'select', 'Open')]

    def on_mount(self):
        self.border_title = 'Home Menu'
        self.border_subtitle = '↑ ↓ choose   ↵ open'


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
        prompt = Text(f'[{"✓" if item[1] in self.selected else " "}] ', style=label.style,
                      no_wrap=True, overflow='ellipsis')
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
        with Vertical(id='value-dialog'):
            yield Static(self.key, markup=False)
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
