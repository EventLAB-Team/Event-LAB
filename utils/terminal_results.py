"""Keyboard and mouse browser for saved experiment results."""
import asyncio

from rich.text import Text
from textual import on, work
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, OptionList, Select, Static

from utils.results import discover_results, export_plot, preview
from utils.terminal_theme import eyebrow


class ResultList(OptionList):
    BINDINGS = [Binding('down', 'cursor_down', 'Experiment', key_display='↑/↓'),
                Binding('enter', 'select', 'Inspect')]


class ResultsBrowser(Vertical):
    DEFAULT_CSS = """
    ResultsBrowser { height: 1fr; }
    #result-list { height: 5; border: solid $rule; padding: 0 1; border-title-color: $muted; border-title-align: left; }
    #result-list:focus { border: solid $cyan 80%; border-title-color: $foreground; }
    #result-controls { height: 3; align-vertical: middle; }
    #result-kind { width: 24; height: 1; margin-right: 1; }
    #result-controls Button { width: auto; min-width: 8; padding: 0 1; }
    #result-info { height: auto; max-height: 3; color: $muted; }
    #result-preview { height: 1fr; border: solid $rule; padding: 0 1; }
    #result-preview:focus { border: solid $cyan 80%; }
    #result-plot { height: auto; }
    """

    def __init__(self, root, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.entries = []

    def compose(self):
        yield ResultList(id='result-list')
        with Horizontal(id='result-controls'):
            yield Select([('Recall@K', 'recall'), ('Similarity matrix', 'matrix'), ('PR curve', 'pr')],
                         value='recall', allow_blank=False, compact=True, id='result-kind')
            yield Button('Open plot', id='open-result', variant='primary', disabled=True)
            yield Button('Refresh', id='refresh-results')
        yield Static('Choose a saved experiment.', id='result-info', markup=False)
        with VerticalScroll(id='result-preview'):
            yield Static('', id='result-plot')

    @work(exclusive=True, group='results-load')
    async def refresh_results(self):
        self.query_one('#result-info', Static).update('Loading saved results…')
        try:
            entries, warnings = await asyncio.to_thread(discover_results, self.root)
        except Exception as error:
            self.query_one('#result-info', Static).update(f'Could not load results: {error}')
            return
        self.entries = entries
        listing = self.query_one('#result-list', OptionList)
        listing.clear_options()
        listing.add_options([Text(e['label'], no_wrap=True, overflow='ellipsis') for e in entries])
        listing.border_title = f'{eyebrow("Saved experiments")}  [#626368]{len(entries):02d}[/]'
        self.query_one('#open-result', Button).disabled = not entries
        if entries:
            listing.highlighted = 0
            self.show_result()
        else:
            self.query_one('#result-info', Static).update('No saved experiments yet. Run an experiment, then return here.')
            self.query_one('#result-plot', Static).update('')
        for warning in warnings:
            self.notify(warning, severity='warning')

    @on(OptionList.OptionHighlighted, '#result-list')
    @on(Select.Changed, '#result-kind')
    def changed(self):
        self.show_result()

    @on(OptionList.OptionSelected, '#result-list')
    def inspect_result(self):
        self.query_one('#result-preview').focus()

    def selected(self):
        index = self.query_one('#result-list', OptionList).highlighted
        return self.entries[index] if index is not None and index < len(self.entries) else None

    @work(exclusive=True, group='results-preview')
    async def show_result(self):
        entry = self.selected()
        if entry is None:
            return
        kind = self.query_one('#result-kind', Select).value
        container = self.query_one('#result-preview')
        self.query_one('#result-info', Static).update(
            entry['label'] + '\n' + entry['source'] + ' · ' + str(entry['metrics'].get('timestamp_utc', '')))
        try:
            content = await asyncio.to_thread(preview, entry, kind,
                max(16, container.size.width - 12), max(4, container.size.height - 6))
        except Exception as error:
            content = Text(f'Cannot preview this result: {error}')
        self.query_one('#result-plot', Static).update(content)
        container.scroll_home(animate=False)

    def on_resize(self):
        if self.entries:
            self.show_result()

    @on(Button.Pressed, '#refresh-results')
    def refresh_clicked(self):
        self.refresh_results()

    @on(Button.Pressed, '#open-result')
    @work(exclusive=True, group='results-export')
    async def open_plot(self):
        entry = self.selected()
        if entry is None:
            return
        try:
            path = await asyncio.to_thread(export_plot, entry, self.query_one('#result-kind', Select).value)
            self.app.open_url(path.resolve().as_uri())
            self.notify(f'Plot: {path}')
        except Exception as error:
            self.notify(f'Could not open plot: {error}', severity='error')
