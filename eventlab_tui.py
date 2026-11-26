"""EventLAB - Localization's keyboard-friendly experiment workbench."""
import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import threading
import time

from rich.text import Text
from textual import on, work
from textual.binding import Binding
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, Footer, Input, RichLog,
                             Select, Static, Link, DataTable, OptionList, TabbedContent,
                             TabPane)

from utils.configuration import (BASELINE_CHOICES, CONFIG, FIELDS, ROOT, dumps, parse_value,
                                 read_config, update_config, write_config,
                                 REPRESENTATION_DOCS)
from utils.experiments import Queue, catalog, expand_batch
from utils.terminal_output import TerminalOutput
from utils.terminal_widgets import (SelectionList, RoleTable, ValueEditor, CompactTable, HomeMenu, EventLogo,
                                    EventField, Panel, PixelView, SectionTabs)
from utils.live_vpr import (LIVE_DATASET, LIVE_METHODS, LiveWorker, ReferenceView, clean_name, delete_recording,
                            recording_path, recordings, register, render)
from utils.terminal_theme import COMPLETE, CYAN, FAINT, MISSING, MUTED, PARTIAL, eventlab_theme, eyebrow
from utils.terminal_results import ResultsBrowser

REPOSITORY = 'https://github.com/EventLAB-Team/Event-LAB'
PAPER = 'https://arxiv.org/abs/2509.14516'
DATASET_LABELS = {'brisbane_event': 'Brisbane Event', 'gardens_point_event': 'Gardens Point',
                  'fast_slow': 'Fast & Slow', 'nsavp': 'NSAVP', 'qcr_event': 'QCR Event',
                  'springfield_event': 'Springfield', 'nyc_event': 'NYC Event', LIVE_DATASET: 'Live recordings'}


class Confirm(ModalScreen[bool]):
    BINDINGS = [('escape', 'dismiss(False)', 'Back')]

    def __init__(self, message, accept='Continue'):
        super().__init__()
        self.message, self.accept = message, accept

    def compose(self):
        with Vertical(id='dialog'):
            with VerticalScroll():
                yield Static(self.message, markup=False)
            with Horizontal(classes='buttons'):
                yield Button(self.accept, id='yes', variant='primary')
                yield Button('Back', id='no')

    def on_mount(self):
        self.query_one('#dialog').border_title = eyebrow('Review & run' if self.accept.startswith('Run ') else 'Confirm')
        self.query_one('#yes').focus()

    @on(Button.Pressed)
    def answer(self, event):
        self.dismiss(event.button.id == 'yes')

def value_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return dumps(value).strip().removesuffix('\n...')


def run_summary(jobs, config):
    """Group repeated combinations, keeping the confirmation short."""
    groups = {}
    for job in jobs:
        key = (job['dataset'], job['reference'])
        queries, baselines = groups.setdefault(key, ([], []))
        if job['query'] not in queries:
            queries.append(job['query'])
        if job['baseline'] not in baselines:
            baselines.append(job['baseline'])
    lines = [f"{len(jobs)} experiment{'s' if len(jobs) != 1 else ''} · "
             f"{len(groups)} reference/query group{'s' if len(groups) != 1 else ''}", '']
    for (dataset, reference), (queries, baselines) in groups.items():
        lines += [dataset, f"  {reference} → {', '.join(queries)}",
                  f"  Baselines: {', '.join(baselines)}", '']
    values = config.get('timewindows', [])
    lines.append(f"{config.get('representation', 'Events')} · Time windows (ms): {', '.join(str(v) for v in values)}")
    overridden = sum(bool(j['config'] or j['baseline_config']) for j in jobs)
    if overridden:
        lines.append(f'Batch overrides apply to {overridden} experiments.')
    lines += ['Saved settings only · continue after failures', 'Results: output/ · detailed logs: logs/']
    return '\n'.join(lines)


class EventLabApp(App):
    ENABLE_COMMAND_PALETTE = False
    TITLE = 'EventLAB - Localization'
    SUB_TITLE = 'Experiment workbench'
    CSS = """
    Screen { background: $background; color: $foreground; }
    * { scrollbar-size-vertical: 1; scrollbar-size-horizontal: 1; }
    #topbar { height: 2; padding: 0 1; background: $background; }
    #home-back { width: 10; min-width: 10; height: 2; margin: 0; padding: 0 1; background: $background;
                 color: $muted; border-bottom: solid $rule; text-style: none; content-align: left top; }
    #home-back:hover { color: $foreground; background: $background; }
    #home-back:focus { color: $primary; background: $background; text-style: bold; }
    #section-tabs { width: 1fr; height: 2; }
    TabbedContent { height: 1fr; }
    TabbedContent > ContentTabs { display: none; }
    TabbedContent > ContentSwitcher { height: 1fr; }
    TabPane { height: 1fr; padding: 0 2; overflow-y: auto; }

    #home-scroll { align: center middle; background: $background; }
    #welcome { width: 100%; max-width: 72; height: auto; }
    #brand { height: 11; width: 100%; }
    #intro { height: auto; margin: 1 0; color: $dim; }
    #home-menu { height: 14; background: $elevated; border: solid $rule-strong; padding: 0 1;
                 border-title-color: $faint; border-title-align: left;
                 border-subtitle-color: $faint; border-subtitle-align: right; }
    #home-menu:focus { border: solid $rule-strong; border-title-color: $cyan; }
    #home-menu > .option-list--option { padding: 0 1; }
    #home-menu > .option-list--option-highlighted { background: $raised; color: $foreground; }
    #home-menu:focus > .option-list--option-highlighted { background: $magenta-deep; color: $primary; }
    #home-menu > .option-list--option-hover { background: $raised; }
    #credits { height: auto; color: $faint; margin-top: 1; }
    .compact #intro { margin: 0; }
    .compact #credits { margin-top: 0; }

    .muted { color: $muted; height: auto; }
    .workspace-intro { height: 3; padding-top: 1; color: $muted; }
    .buttons { height: 3; min-height: 3; align-vertical: middle; }
    Button { height: 1; min-height: 1; border: none; background: $elevated; color: $dim;
             margin-right: 1; text-style: none; }
    Button:hover { background: $raised; color: $foreground; }
    Button:focus { background: $magenta-deep; color: $primary; text-style: bold; }
    Button.-primary { background: $primary; color: $background; text-style: bold; }
    Button.-primary:hover { background: $primary-lighten-1; color: $background; }
    Button.-primary:focus { background: $foreground; color: $background; text-style: bold; }
    Button.-error { background: $elevated; color: $error; }
    Button:disabled, Button.-primary:disabled, Button.-error:disabled { background: $elevated; color: $faint; text-opacity: 100%; }
    .buttons Button { min-width: 8; width: auto; padding: 0 2; }
    .selection-all { width: 100%; min-width: 0; margin: 0; background: $background; color: $faint; }
    .selection-all:hover { background: $raised; }
    #download-status { height: 2; color: $muted; }

    .columns { height: 1fr; }
    .column { width: 1fr; margin-right: 1; border: solid $rule; padding: 0 1; background: $background;
              border-title-color: $muted; border-title-align: left; }
    .column:focus-within { border: solid $cyan 80%; border-title-color: $foreground; }
    OptionList, DataTable { background: $background; border: none; }
    OptionList:focus, DataTable:focus { background-tint: $background 0%; }
    OptionList > .option-list--option-highlighted { background: $raised; color: $foreground; }
    OptionList:focus > .option-list--option-highlighted { background: $magenta-deep; color: $primary; text-style: bold; }
    #result-list { border: solid $rule; border-title-color: $muted; border-title-align: left; }
    #result-list:focus { border: solid $cyan 80%; border-title-color: $foreground; }
    SelectionList { height: 1fr; padding: 0; }
    SelectionList > .option-list--option-highlighted { background: $raised; color: $foreground; }
    SelectionList:focus > .option-list--option-highlighted { background: $magenta-deep; color: $primary; text-style: bold; }
    DataTable > .datatable--header { background: $background; color: $faint; text-style: none; }
    DataTable > .datatable--cursor { background: $raised; color: $foreground; }
    DataTable:focus > .datatable--cursor { background: $magenta-deep; color: $primary; text-style: bold; }
    DataTable > .datatable--hover { background: $raised; }
    #trajectory-title { height: 1; color: $cyan; }
    #fields, #trajectories { height: 1fr; }
    #fields { border: solid $rule; border-title-color: $muted; border-title-align: left; }
    #fields:focus { border: solid $cyan 80%; border-title-color: $foreground; }
    #field-docs { display: none; }
    #field-help { height: auto; min-height: 3; max-height: 5; padding-top: 1; color: $muted; }
    #config-file { height: 3; margin: 1 0 0 0; }
    .compact #config-file { margin: 0; }
    #config-message { display: none; }
    #experiment-preview { height: 2; padding-top: 1; color: $primary; }
    #batch-path { display: none; }
    #batch-label { height: 1; color: $faint; }

    Select > SelectCurrent { background: $background; border: solid $rule; }
    Select:focus > SelectCurrent { border: solid $cyan 80%; background: $background; }
    Select > SelectCurrent Static#label { color: $foreground; }
    Select > SelectCurrent .arrow { color: $faint; }
    Select > SelectOverlay { background: $elevated; border: solid $rule-strong; }
    Select.-textual-compact > SelectCurrent { border: none; background: $elevated; }
    Input { background: $background; border: solid $rule; }
    Input:focus { border: solid $cyan 80%; background: $background; }
    Input.-textual-compact { background: $elevated; border: none; }
    Link { color: $cyan; }

    #log { height: 1fr; border: solid $rule; background: $background; border-title-color: $muted; border-title-align: left; }
    #log:focus { border: solid $cyan 80%; border-title-color: $foreground; }
    #progress { height: 1; color: $cyan; text-wrap: nowrap; text-overflow: ellipsis; }
    #run-status { height: auto; margin-top: 1; color: $primary; }
    Footer { background: $background; color: $muted; }
    Toast { background: $elevated; }
    Tooltip { background: $elevated; color: $dim; border: solid $rule-strong; }

    Confirm, ValueEditor { align: center middle; background: $background 75%; }
    #dialog { width: 80%; max-width: 100; height: auto; max-height: 80%; border: solid $rule-strong;
              background: $elevated; padding: 1 2; border-title-color: $primary; border-title-style: bold; border-title-align: left; }
    #dialog > VerticalScroll { height: auto; max-height: 1fr; }
    #live-side { width: 34; }
    #live-side Input { margin-bottom: 1; }
    #live-side Button { width: 100%; margin: 0 0 1 0; }
    #live-references { height: 1fr; border: solid $rule; }
    #live-references:focus { border: solid $cyan 80%; }
    #live-method { margin-bottom: 1; }
    #live-ref-buttons { height: 1; margin-bottom: 1; }
    #live-ref-buttons Button { width: 1fr; margin: 0 1 0 0; }
    #live-camera-status, #live-match-status { height: 2; color: $muted; }
    #live-progress { height: auto; color: $primary; }
    #value-dialog { width: 64; max-width: 95%; height: auto; border: solid $rule-strong; padding: 1 2;
                    background: $elevated; border-title-color: $cyan; border-title-align: left; }
    """
    BINDINGS = [Binding('h', 'home', 'Home'), Binding('x', 'quit', 'Quit'),
                Binding('s', 'save_config', 'Save settings'), Binding('u', 'reload', 'Reload'),
                Binding('v', 'review', 'Review & run'),
                Binding('escape', 'home', 'Home', show=False),
                Binding('ctrl+q', 'quit', 'Quit', show=False),
                Binding('ctrl+r', 'reload', 'Reload', show=False),
                Binding('ctrl+left', 'section(-1)', 'Previous section', show=False),
                Binding('ctrl+right', 'section(1)', 'Next section', show=False),
                Binding('ctrl+s', 'save_config', 'Save settings', show=False),
                Binding('f1', 'home', 'Home', show=False), Binding('f5', 'review', 'Review & run', show=False)]

    def __init__(self, config_path=CONFIG):
        super().__init__()
        self.config_path = Path(config_path)
        self.baselines, self.datasets = catalog(read_config(self.config_path))
        self.datasets.pop('qut_event_walking', None)
        self.edit_path = self.config_path
        self.edit_data = {}
        self.pending = {}
        self.sequence_choices = {}
        self.imported_batch = None
        self.queue = Queue(output=self.append_output)
        self.terminal_output = TerminalOutput()
        self.running = False
        self.active_dataset = None
        self.log_buffer = []
        self.animation_level = "none"
        self.layout_size = None
        self.download_states = {}
        self.download_root = None
        self.download_selection = set()
        self.download_dataset = next(iter(self.download_datasets), None)
        self.download_config_stamp = None
        # Live section: the camera thread owns the device and the window hand-off; the UI only posts
        # requests to it (`record_request`), which it applies by reopening the camera.
        self.camera_stop = threading.Event()
        self.camera_running = False
        self.record_request = None
        self.recording_name = None
        self.live_pending = None
        self.live_places = None
        self.live_windows = None
        self.review, self.review_name, self.review_index = None, None, 0
        self.localiser = None
        self.live_reference = None
        self.reference_view = None
        self.live_place = None
        self.register_theme(eventlab_theme())
        self.theme = 'eventlab'

    def compose(self) -> ComposeResult:
        with Horizontal(id='topbar'):
            yield Button('‹ home', id='home-back')
            yield SectionTabs(id='section-tabs')
        with TabbedContent(initial='home', id='tabs'):
            with TabPane('Home', id='home'):
                with EventField(id='home-scroll'):
                    with Vertical(id='welcome'):
                        yield EventLogo(id='brand')
                        yield Static('EventLAB - Localization: a keyboard workbench for benchmarking '
                                     'event-based place recognition.', id='intro')
                        yield HomeMenu(
                            HomeMenu.item(1, 'Run an experiment', 'Choose methods, datasets and trajectories.'),
                            HomeMenu.item(2, 'Manage downloads', 'Prepare datasets and baseline model weights.'),
                            HomeMenu.item(3, 'Edit configuration', 'Review runtime and baseline configuration.'),
                            HomeMenu.item(4, 'View activity', 'Follow progress and inspect experiment logs.'),
                            HomeMenu.item(5, 'View results', 'Explore matrices, PR curves and Recall@K.'),
                            HomeMenu.item(6, 'Live localisation', 'Record references and localise with a camera.'),
                            id='home-menu')
                        yield Static(Text.assemble(
                            ('Towards Standardized Evaluation of Neuromorphic Localization Methods\n', MUTED),
                            ('Hines · Fontan · Milford · Fischer  ·  EventLAB Team\n', FAINT),
                            ('↗ ', FAINT), (REPOSITORY.removeprefix('https://'), f'{CYAN} link {REPOSITORY}'),
                            ('   ↗ ', FAINT), (PAPER.removeprefix('https://'), f'{CYAN} link {PAPER}')),
                            id='credits')
            with TabPane('Configuration', id='configuration'):
                yield Select([('Runtime · config.yaml', str(self.config_path))] +
                             [(f'Baseline · {name}', str(ROOT / 'baselines' / f'{name}.yaml')) for name in self.baselines],
                             value=str(self.config_path), allow_blank=False, id='config-file')
                yield Static('Edit settings, then Save. Each method owns its event input format.', id='config-message')
                yield CompactTable(id='fields', cursor_type='row')
                yield Static('', id='field-help', markup=False)
                yield Link('reader.with_repr() ↗', url=REPRESENTATION_DOCS, id='field-docs')
                with Horizontal(classes='buttons'):
                    yield Button('Save settings', id='save-config', variant='primary')
                    yield Button('Revert', id='revert-config')
            with TabPane('Experiments', id='experiments'):
                yield Static('Choose what to compare. Assign one reference (R) and one or more queries (Q) per dataset.', classes='workspace-intro')
                with Horizontal(classes='columns'):
                    with Panel(1, 'Methods', classes='column'):
                        yield SelectionList(*[(n, n) for n in self.baselines], id='baselines')
                        yield Button('Select all', id='all-baselines', classes='selection-all')
                    with Panel(2, 'Datasets', classes='column'):
                        yield SelectionList(*[(DATASET_LABELS.get(n, n), n)
                                             for n in self.datasets], id='datasets')
                        yield Button('Select all', id='all-datasets', classes='selection-all')
                    with Panel(3, 'Trajectories', classes='column'):
                        yield Static('Highlight a dataset', id='trajectory-title', markup=False)
                        yield RoleTable(id='trajectories', cursor_type='row')
                        yield Static('R ref · Q query\nSpace cycle · Del off', classes='muted')
                        yield Button('Select all', id='all-trajectories', classes='selection-all',
                                     tooltip='Keep the reference (or use the first trajectory); select all others as queries.')
                yield Static('No experiments selected.', id='experiment-preview', markup=False)
                yield Input(str(ROOT / 'batch_config.yaml'), id='batch-path')
                yield Static('Batch file: batch_config.yaml', id='batch-label', markup=False)
                with Horizontal(classes='buttons'):
                    yield Button('File…', id='choose-batch')
                    yield Button('Load', id='load-batch')
                    yield Button('Save', id='save-batch')
                    yield Button('New', id='new-batch')
                    yield Button('Review & run', id='run', variant='primary')
            with TabPane('Downloads', id='downloads'):
                yield Static('Prepare repositories, configured weights, raw recordings and ground truth. No inference or frame generation.', classes='workspace-intro')
                with Horizontal(classes='columns'):
                    with Panel(1, 'Baselines', classes='column', id='download-methods'):
                        yield SelectionList(*[(n, n) for n in self.baselines], id='download-baselines')
                        yield Button('Select all', id='all-download-baselines', classes='selection-all')
                    with Panel(2, 'Datasets', classes='column'):
                        yield SelectionList(*[(name, name) for name in self.download_datasets],
                                            id='download-datasets')
                        yield Button('Select all', id='all-download-datasets', classes='selection-all')
                    with Panel(3, 'Recordings', classes='column'):
                        yield SelectionList(id='download-sequences')
                        yield Button('Select all', id='all-download-sequences', classes='selection-all')
                yield Static('Choose datasets or individual recordings to download.', id='download-status', markup=False)
                with Horizontal(classes='buttons'):
                    yield Button('Download selected', id='download', variant='primary')
                    yield Button('Check files', id='check-downloads', tooltip='Recheck availability without selecting missing downloads.')
            with TabPane('Run logs', id='logs'):
                yield Static('Ready. Experiments and downloads run sequentially.', id='run-status', markup=False)
                yield Static('', id='progress', markup=False)
                yield RichLog(id='log', wrap=True, highlight=False, max_lines=4000)
                with Horizontal(classes='buttons'):
                    yield Button('Cancel queue', id='cancel', variant='error', disabled=True)
            with TabPane('Results', id='results'):
                yield ResultsBrowser(ROOT / 'output')
            with TabPane('Live', id='live'):
                yield Static('Record a reference with the camera, then localise against it: the method maps the '
                             'reference once, then matches every live window. No ground truth.', classes='workspace-intro')
                with Horizontal(classes='columns'):
                    with Panel(1, 'Camera', classes='column'):
                        yield PixelView(id='live-view')
                        yield Static('Camera off', id='live-camera-status', markup=False)
                    with Panel(2, 'Closest place', classes='column'):
                        yield PixelView(id='live-match')
                        yield Static('Choose a reference and a method, then Localise.', id='live-match-status',
                                     markup=False)
                    with Panel(3, 'Session', classes='column', id='live-side'):
                        yield Input(placeholder='Reference name', id='live-name')
                        yield Button('● Record', id='live-record', variant='primary')
                        yield OptionList(id='live-references')
                        with Horizontal(id='live-ref-buttons'):
                            yield Button('View', id='live-view-ref',
                                         tooltip='Scrub through the recording: ←→ window, shift ←→ second')
                            yield Button('Delete', id='live-delete-ref', variant='error')
                        yield Select([(name, name) for name in LIVE_METHODS], value=next(iter(LIVE_METHODS)),
                                     allow_blank=False, id='live-method')
                        yield Input(placeholder='Places: every window', id='live-places', type='integer',
                                    tooltip='Map only this many places, evenly spaced in time through the '
                                            'reference. Blank: every window is a place. Applies at Localise.')
                        yield Button('Localise', id='live-localise')
                        yield Static('', id='live-progress', markup=False)
        yield Footer()

    @property
    def download_datasets(self):
        """Datasets that can be downloaded: everything but the live recordings, which are local."""
        return {name: sequences for name, sequences in self.datasets.items() if name != LIVE_DATASET}

    async def on_mount(self):
        await self.load_fields(self.edit_path)
        self.query_one('#trajectories', DataTable).add_columns('Trajectory', 'Role')
        self.progress_display = self.query_one('#progress', Static)
        self.log_display = self.query_one('#log', RichLog)
        self.last_progress = ''
        self.set_interval(.1, self.flush_logs)
        self.set_interval(1, self.check_download_config)
        self.set_interval(.05, self.show_live_frame)
        self.query_one('#fields').border_title = eyebrow('Settings')
        self.log_display.border_title = eyebrow('Output')
        self.query_one('#home-menu').focus()
        self.query_one('#topbar').display = False

    def on_resize(self, event):
        size = (event.size.width < 70, event.size.height < 28)
        if size == self.layout_size:
            return
        self.layout_size = size
        self.screen.set_class(size[1], 'compact')
        for columns in self.query('.columns'):
            columns.styles.layout = 'vertical' if size[0] else 'horizontal'
            columns.styles.height = 'auto' if size[0] else '1fr'
            for column in columns.query('.column'):
                column.styles.height = 12 if size[0] else '1fr'
                column.styles.width = '1fr'

    async def load_fields(self, path):
        data = read_config(path)
        self.edit_path, self.edit_data = Path(path), data
        self.pending = copy.deepcopy(data)
        table = self.query_one('#fields', DataTable)
        table.clear(columns=True)
        table.add_columns('Setting', 'Value')
        self.field_keys = [k for k in data if self.edit_path != self.config_path or k in FIELDS]
        table.add_rows([(key, self.field_value(key, data[key])) for key in self.field_keys])

    def field_choices(self, key):
        if self.edit_path == self.config_path:
            return FIELDS[key][2]
        return BASELINE_CHOICES.get(self.edit_path.stem, {}).get(key)

    def field_value(self, key, value):
        if self.field_choices(key):
            return f'{value_text(value)} ▾'
        return value_text(value)

    def field_description(self, key):
        if self.edit_path == self.config_path:
            return FIELDS[key][3]
        comment = self.edit_data.ca.items.get(key)
        return comment[2].value.strip().lstrip('#').strip() if comment and comment[2] else ''

    def dirty(self):
        return self.pending != self.edit_data

    @on(DataTable.RowHighlighted, '#fields')
    def field_highlighted(self, event):
        if event.cursor_row >= len(self.field_keys):
            return
        key = self.field_keys[event.cursor_row]
        help_text = self.field_description(key)
        self.query_one('#field-docs', Link).display = False
        if self.field_choices(key):
            hint = 'Enter open dropdown · S save'
        elif key == 'data_path' and self.edit_path == self.config_path:
            hint = 'Enter edit and save dataset folder'
        else:
            hint = 'Enter edit · Space toggle · S save'
        self.query_one('#field-help', Static).update(hint + '\n' + help_text)

    @on(DataTable.RowSelected, '#fields')
    def edit_field(self):
        index = self.query_one('#fields', DataTable).cursor_row
        key = self.field_keys[index]
        runtime = self.edit_path == self.config_path
        value = self.pending[key]
        def accept(text):
            if text is None:
                return
            try:
                parsed = parse_value(key, text, value, runtime)
                if runtime and key == 'data_path':
                    # The folder drives both preparation and availability checks.
                    # Commit it on Enter so neither can silently use the old path.
                    update_config(self.config_path, {key: parsed})
                    self.edit_data[key] = parsed
                    self.call_after_refresh(self.refresh_downloads)
                    self.notify('Dataset folder saved')
                self.pending[key] = parsed
                self.query_one('#fields', DataTable).update_cell_at((index, 1), self.field_value(key, self.pending[key]))
            except Exception as error:
                self.notify(str(error), severity='error')
        if isinstance(value, bool):
            accept('false' if value else 'true')
        else:
            choices = self.field_choices(key)
            self.push_screen(ValueEditor(
                key, value_text(value), choices, self.field_description(key)), accept)

    @on(Select.Changed, '#config-file')
    async def change_config(self, event):
        if not self.is_mounted or event.value == Select.NULL or Path(str(event.value)) == self.edit_path:
            return
        path = Path(str(event.value))
        if self.dirty():
            self.query_one('#config-file', Select).value = str(self.edit_path)
            self.push_screen(Confirm('Discard unsaved configuration changes?', 'Discard'),
                             lambda accepted: self.switch_file(path) if accepted else None)
        else:
            await self.load_fields(path)

    @work
    async def switch_file(self, path):
        await self.load_fields(path)
        self.query_one('#config-file', Select).value = str(path)

    def action_save_config(self):
        try:
            changes = {key: value for key, value in self.pending.items() if value != self.edit_data.get(key)}
            update_config(self.edit_path, changes, self.edit_path == self.config_path)
            self.edit_data = read_config(self.edit_path)
            self.notify(f'Saved {self.edit_path.name}')
            if self.edit_path == self.config_path and self.download_root is not None:
                self.refresh_downloads()
        except Exception as error:
            self.notify(str(error), severity='error', timeout=8)

    def action_reload(self):
        if self.dirty():
            self.push_screen(Confirm('Discard pending edits and reload?', 'Reload'),
                             lambda yes: self.switch_file(self.edit_path) if yes else None)
        else:
            self.switch_file(self.edit_path)

    def action_home(self):
        self.query_one('#tabs', TabbedContent).active = 'home'

    def action_quit(self):
        if self.running or self.dirty():
            message = ('Stop the active queue and quit?' if self.running else 'Discard unsaved changes and quit?')
            self.push_screen(Confirm(message, 'Quit'), self.quit_confirmed)
        else:
            self.exit()

    @work
    async def quit_confirmed(self, accepted):
        if accepted:
            await self.queue.cancel()
            self.exit()

    @on(OptionList.OptionHighlighted, '#datasets')
    def dataset_highlighted(self, event):
        self.show_dataset(self.query_one('#datasets', SelectionList).items[event.option_index][1])

    def show_dataset(self, name):
        self.active_dataset = name
        self.query_one('#trajectory-title', Static).update(name)
        reference, queries = self.sequence_choices.get(name, (None, []))
        table = self.query_one('#trajectories', DataTable)
        table.clear()
        table.add_rows([(seq, 'R' if seq == reference else 'Q' if seq in queries else '') for seq in self.datasets[name]])
        self.update_role_selection_button()

    def update_role_selection_button(self):
        reference, queries = self.sequence_choices.get(self.active_dataset, (None, []))
        all_selected = reference and len(queries) + 1 == len(self.datasets[self.active_dataset])
        self.query_one('#all-trajectories', Button).label = 'Unselect all' if all_selected else 'Select all'

    def action_toggle_all_roles(self):
        name = self.active_dataset
        if name is None or self.imported_batch is not None:
            return
        sequences = self.datasets[name]
        reference, queries = self.sequence_choices.get(name, (None, []))
        if reference and len(queries) + 1 == len(sequences):
            self.sequence_choices[name] = (None, [])
        else:
            reference = reference or sequences[0]
            self.sequence_choices[name] = (reference, [seq for seq in sequences if seq != reference])
            self.query_one('#datasets', SelectionList).select(name)
        self.show_dataset(name)
        self.preview()

    @on(DataTable.RowSelected, '#trajectories')
    def role_clicked(self):
        self.set_role('cycle')

    def set_role(self, role):
        name = self.active_dataset
        if name is None or self.imported_batch is not None:
            return
        table = self.query_one('#trajectories', DataTable)
        seq = self.datasets[name][table.cursor_row]
        reference, queries = self.sequence_choices.get(name, (None, []))
        current = 'R' if seq == reference else 'Q' if seq in queries else ''
        if role == 'cycle':
            role = {'': 'Q' if reference else 'R', 'R': '', 'Q': 'R'}[current]
        queries = [q for q in queries if q != seq]
        if reference == seq:
            reference = None
        if role == 'R':
            reference = seq
        elif role == 'Q':
            queries.append(seq)
        self.sequence_choices[name] = (reference, queries)
        if role in ('R', 'Q'):
            self.query_one('#datasets', SelectionList).select(name)
        self.update_role_selection_button()
        for index, trajectory in enumerate(self.datasets[name]):
            table.update_cell_at((index, 1), 'R' if trajectory == reference else 'Q' if trajectory in queries else '')
        self.preview()

    @on(SelectionList.SelectedChanged)
    def selection_changed(self, event):
        listing = event.selection_list
        if listing.id == 'download-datasets':
            for name, sequences in self.download_datasets.items():
                keys = {(name, seq) for seq in sequences}
                if name in listing.selected:
                    self.download_selection.update(keys)
                elif keys <= self.download_selection:
                    self.download_selection.difference_update(keys)
            self.render_download_groups()
            self.render_download_recordings()
        elif listing.id == 'download-sequences':
            self.download_selection.difference_update({(self.download_dataset, seq) for seq in self.download_datasets[self.download_dataset]})
            self.download_selection.update(listing.selected)
            self.render_download_groups()
        self.query_one('#all-' + listing.id, Button).label = (
            'Unselect all' if len(listing.selected) == len(listing.items) else 'Select all')
        if event.selection_list.id in ('baselines', 'datasets'):
            self.preview()

    def document(self):
        if self.imported_batch is not None:
            return copy.deepcopy(self.imported_batch)
        selected = self.query_one('#baselines', SelectionList).selected
        entries = []
        for name in self.query_one('#datasets', SelectionList).selected:
            reference, queries = self.sequence_choices.get(name, (None, []))
            entries.append(dict(dataset=name, reference=reference, queries=list(queries), baselines=list(selected)))
        return {'batch_experiments': entries}

    def preview(self):
        entries = self.document()['batch_experiments']
        message = 'Select baselines and datasets'
        count = 0
        for entry in entries:
            if not entry['reference']:
                message = f"{entry['dataset']}: choose a reference (R)"
                break
            if not entry['queries']:
                message = f"{entry['dataset']}: choose at least one query (Q)"
                break
            count += len(entry['baselines']) * len(entry['queries'])
        else:
            if count:
                message = f'{count} experiments ready · current saved settings · continue after failures'
        self.query_one('#experiment-preview', Static).update(message)

    def action_review(self):
        if self.running:
            self.notify('A queue is already running.', severity='warning')
            return
        try:
            document = self.document()
            config = read_config(self.config_path)
            jobs = expand_batch(document, config)
            message = run_summary(jobs, config)
            self.push_screen(Confirm(message, f"Run {len(jobs)} experiment{'s' if len(jobs) != 1 else ''}"),
                             lambda accepted: self.call_after_refresh(self.run_experiments, document) if accepted else None)
        except Exception as error:
            self.notify(str(error), severity='error')

    def append_log(self, text):
        self.log_buffer.extend(self.terminal_output.finish())
        self.log_buffer.append(text)
        if text.startswith('[') or text.startswith('Queue finished'):
            self.query_one('#run-status', Static).update(text)

    def append_output(self, text):
        self.log_buffer.extend(self.terminal_output.feed(text))

    def flush_logs(self):
        if not self.progress_display.is_mounted:
            return
        if self.last_progress != self.terminal_output.progress:
            self.last_progress = self.terminal_output.progress
            self.progress_display.update(self.last_progress)
        if self.log_buffer:
            self.log_display.write(Text.from_ansi('\n'.join(self.log_buffer)))
            self.log_buffer.clear()

    @on(TabbedContent.TabActivated)
    def section_changed(self, event):
        targets = {'home': '#home-menu', 'configuration': '#fields', 'experiments': '#baselines',
                   'downloads': '#download-baselines', 'logs': '#log', 'results': '#result-list',
                   'live': '#live-name'}
        if event.pane.id == 'results':
            self.query_one(ResultsBrowser).refresh_results()
        elif event.pane.id == 'downloads':
            self.refresh_downloads()
        if event.pane.id == 'live':
            self.refresh_references()
            self.start_camera()
        elif not (self.recording_name or self.localiser):
            self.camera_stop.set()  # leave the device free unless the Live section is using it
        self.query_one(SectionTabs).show(event.pane.id)
        self.query_one('#topbar').display = event.pane.id != 'home'
        for animated in (self.query_one(EventLogo), self.query_one(EventField)):
            if hasattr(animated, 'animation'):
                if event.pane.id == 'home':
                    animated.animation.resume()
                else:
                    animated.animation.pause()
        matches = list(self.query(targets[event.pane.id]))
        if matches:
            matches[0].focus()

    @on(OptionList.OptionSelected, '#home-menu')
    def home_selected(self, event):
        self.query_one('#tabs', TabbedContent).active = ['experiments', 'downloads', 'configuration', 'logs', 'results',
                                                         'live'][event.option_index]

    def check_action(self, action, parameters):
        if not self.is_mounted:
            return True
        if action in ('home', 'quit', 'save_config', 'reload', 'review') and (
                len(self.screen_stack) > 1 or isinstance(self.focused, (Input, Select))):
            return False
        section = self.query_one('#tabs', TabbedContent).active
        if action == 'home' and section == 'home':
            return False
        if action == 'home' and section == 'live' and self.review is not None:
            return False  # Esc closes the recording viewer instead
        if action in ('save_config', 'reload') and section != 'configuration':
            return False
        if action == 'review' and section != 'experiments':
            return False
        return True

    def action_pane(self, direction):
        section = self.query_one('#tabs', TabbedContent).active
        ids = {'experiments': ['baselines', 'datasets', 'trajectories'],
               'downloads': ['download-baselines', 'download-datasets', 'download-sequences'],
               'configuration': ['fields'], 'home': ['home-menu'], 'logs': ['log'],
               'results': ['result-list', 'result-kind', 'result-preview'],
               'live': ['live-name', 'live-references', 'live-method', 'live-places']}[section]
        available = [self.query_one('#' + name) for name in ids if not self.query_one('#' + name).disabled]
        if available:
            index = available.index(self.focused) if self.focused in available else (-1 if direction > 0 else 0)
            available[(index + direction) % len(available)].focus()

    def action_section(self, direction):
        tabs = self.query_one('#tabs', TabbedContent)
        names = ['home', 'configuration', 'experiments', 'downloads', 'logs', 'results', 'live']
        tabs.active = names[(names.index(tabs.active) + direction) % len(names)]

    def on_key(self, event):
        if len(self.screen_stack) > 1 or isinstance(self.focused, (Input, Select)):
            return
        if self.query_one('#tabs', TabbedContent).active == 'live' and self.live_key(event.key):
            event.stop()
            event.prevent_default()
            return
        if self.query_one('#tabs', TabbedContent).active == 'home' and event.key in ('1', '2', '3', '4', '5', '6'):
            self.query_one('#tabs', TabbedContent).active = ['experiments', 'downloads', 'configuration', 'logs', 'results',
                                                             'live'][int(event.key)-1]
            event.stop()
        elif event.key in ('left', 'right'):
            self.action_pane(1 if event.key == 'right' else -1)
            event.stop()
            event.prevent_default()
        elif event.key == 'space' and self.focused is self.query_one('#fields'):
            self.edit_field()
            event.stop()

    def set_running(self, value):
        if not value:
            self.log_buffer.extend(self.terminal_output.finish())
            self.flush_logs()
        self.running = value
        if value:
            # Disabling a focused Run button otherwise focuses another experiment
            # control, whose TabPane.Focused message switches us straight back.
            self.log_display.screen.set_focus(None)
            self.query_one('#tabs', TabbedContent).active = 'logs'
        self.query_one('#cancel', Button).disabled = not value
        for name in ('run', 'download'):
            self.query_one(f'#{name}', Button).disabled = value

    @work
    async def run_experiments(self, document):
        self.set_running(True)
        try:
            await self.queue.experiments(document, self.append_log, self.config_path)
        except Exception as error:
            self.append_log(f'Error: {error}')
            self.notify(str(error), severity='error')
        finally:
            self.set_running(False)

    @work
    async def download_selected(self, accepted):
        if not accepted or self.running:
            return
        self.set_running(True)
        try:
            commands = []
            for name in self.query_one('#download-baselines', SelectionList).selected:
                commands.append((name, [sys.executable, '-u', str(ROOT / 'eventlab_prepare.py'),
                                       '--config', str(self.config_path), 'baseline', name], {}))
            for name, sequence in sorted(self.download_selection):
                commands.append((f'{name}/{sequence}', [sys.executable, '-u', str(ROOT / 'eventlab_prepare.py'),
                                 '--config', str(self.config_path), 'dataset', name, sequence], {}))
            await self.queue.run(commands, self.append_log)
        except Exception as error:
            self.append_log(f'Error: {error}')
        finally:
            self.set_running(False)
            self.refresh_downloads()

    @work(exclusive=True, group='download-status')
    async def refresh_downloads(self, reset=False):
        self.download_config_stamp = self.config_path.stat().st_mtime_ns
        self.query_one('#download-status', Static).update('Checking local dataset files…')
        def scan():
            from datasets.prepare import sequence_download_status
            from utils.preparation import baseline_availability
            config = read_config(self.config_path)
            config['data_path'] = str((ROOT / Path(os.path.expandvars(config['data_path'])).expanduser()).resolve())
            states = {}
            for name, sequences in self.download_datasets.items():
                data = read_config(ROOT / 'datasets' / f'{name}.yaml')
                for sequence in sequences:
                    states[(name, sequence)] = sequence_download_status(config, data, name, sequence)
            baselines = {name: baseline_availability(name) for name in self.baselines}
            return config['data_path'], states, baselines
        try:
            root, states, baselines = await asyncio.to_thread(scan)
        except Exception as error:
            self.query_one('#download-status', Static).update(f'Could not check files: {error}')
            return
        for key, status in states.items():
            if status == 'Downloaded' and self.download_states.get(key) != status:
                self.download_selection.discard(key)
        self.download_root, self.download_states = root, states
        listing = self.query_one('#download-baselines', SelectionList)
        for index, name in enumerate(self.baselines):
            available, total, missing = baselines[name]
            color = COMPLETE if available == total else PARTIAL if available else MISSING
            listing.items[index] = (Text(f'{name} ({available}/{total})', style=color), name)
            listing.replace_option_prompt_at_index(index, listing.prompt(listing.items[index]))
        self.baseline_states = baselines
        self.render_download_groups()
        self.render_download_recordings()
        downloaded = sum(status == 'Downloaded' for status in states.values())
        self.query_one('#download-status', Static).update(Text.assemble(
            (f'{downloaded}/{len(states)} available · ', MUTED),
            ('Green: complete', COMPLETE), (' · ', MUTED), ('Orange: partial', PARTIAL),
            (' · ', MUTED), ('Red: missing\n', MISSING),
            (f'Folder: {root}', MUTED)))

    def check_download_config(self):
        try:
            changed = self.config_path.stat().st_mtime_ns != self.download_config_stamp
        except OSError:
            return
        if self.download_root is not None and changed:
            self.refresh_downloads()

    @on(OptionList.OptionHighlighted, '#download-datasets')
    def download_dataset_highlighted(self, event):
        self.download_dataset = list(self.download_datasets)[event.option_index]
        self.render_download_recordings()

    @on(OptionList.OptionHighlighted, '#download-baselines')
    def baseline_highlighted(self, event):
        name = self.baselines[event.option_index]
        state = getattr(self, 'baseline_states', {}).get(name)
        if state:
            available, total, missing = state
            self.query_one('#download-baselines').tooltip = (
                f'{name}: {available}/{total} required assets present. ' +
                ('Missing/unverified: ' + ', '.join(missing) if missing else 'All checked assets available.'))

    def render_download_recordings(self):
        listing = self.query_one('#download-sequences', SelectionList)
        name = self.download_dataset
        items = [(Text(seq, style={'Downloaded': COMPLETE, 'Partial': PARTIAL}.get(
                      self.download_states.get((name, seq)), MISSING)), (name, seq))
                 for seq in self.download_datasets[name]]
        changed_dataset = [i[1] for i in listing.items] != [i[1] for i in items]
        listing.items = items
        listing.selected = [key for _, key in items if key in self.download_selection]
        if changed_dataset:
            listing.clear_options()
            listing.add_options([listing.prompt(item) for item in items])
            listing.highlighted = 0 if items else None
        else:
            for index, item in enumerate(items):
                listing.replace_option_prompt_at_index(index, listing.prompt(item))
        self.query_one('#all-download-sequences', Button).label = 'Unselect all' if len(listing.selected) == len(items) else 'Select all'

    def render_download_groups(self):
        listing = self.query_one('#download-datasets', SelectionList)
        listing.selected = []
        for index, (name, sequences) in enumerate(self.download_datasets.items()):
            selected = sum((name, seq) in self.download_selection for seq in sequences)
            available = sum(self.download_states.get((name, seq)) == 'Downloaded' for seq in sequences)
            partial = available > 0 or any(self.download_states.get((name, seq)) == 'Partial' for seq in sequences)
            color = COMPLETE if available == len(sequences) else PARTIAL if partial else MISSING
            if selected == len(sequences):
                listing.selected.append(name)
            listing.items[index] = (Text(f'{name} ({available}/{len(sequences)})', style=color), name)
            listing.replace_option_prompt_at_index(index, listing.prompt(listing.items[index]))
        self.query_one('#all-download-datasets', Button).label = 'Unselect all' if len(listing.selected) == len(self.download_datasets) else 'Select all'

    @on(Button.Pressed)
    async def button(self, event):
        name = event.button.id
        if name == 'home-back':
            self.action_home()
        elif name == 'all-trajectories':
            self.action_toggle_all_roles()
        elif name and name.startswith('all-'):
            listing = self.query_one('#' + name[4:], SelectionList)
            if not listing.disabled:
                listing.action_toggle_all()
        elif name == 'check-downloads':
            self.refresh_downloads(reset=True)
        elif name == 'choose-batch':
            def chosen(value):
                if value:
                    self.query_one('#batch-path', Input).value = value
                    self.query_one('#batch-label', Static).update('Batch file: ' + value)
            self.push_screen(ValueEditor('Batch file', self.query_one('#batch-path', Input).value), chosen)
        elif name == 'save-config':
            self.action_save_config()
        elif name == 'revert-config':
            await self.load_fields(self.edit_path)
        elif name == 'run':
            self.action_review()
        elif name == 'cancel':
            await self.queue.cancel()
        elif name == 'download':
            baselines = self.query_one('#download-baselines', SelectionList).selected
            sequences = sorted(self.download_selection)
            if not baselines and not sequences:
                self.notify('Select at least one baseline or sequence.', severity='warning')
                return
            message = 'Download code, configured weights, raw data and ground truth:\n\n'
            message += '\n'.join(baselines + [f'{n}/{s}' for n, s in sequences])
            message += '\n\nExisting assets will be reused. No inference will run.'
            self.push_screen(Confirm(message, 'Download'), self.download_selected)
        elif name in ('load-batch', 'save-batch'):
            try:
                path = Path(self.query_one('#batch-path', Input).value).expanduser()
                if name == 'load-batch':
                    document = read_config(path)
                    expand_batch(document, read_config(self.config_path))
                    entries = document['batch_experiments']
                    editable = (all(e['dataset'] in self.datasets for e in entries)
                                and len({e['dataset'] for e in entries}) == len(entries)
                                and all(not e.get('config') and not e.get('baseline_config')
                                        and e['baselines'] == entries[0]['baselines'] for e in entries))
                    self.imported_batch = None if editable else document
                    if editable:
                        self.sequence_choices = {e['dataset']: (e['reference'], list(e['queries'])) for e in entries}
                        self.query_one('#baselines', SelectionList).deselect_all()
                        self.query_one('#datasets', SelectionList).deselect_all()
                        for baseline in entries[0]['baselines']:
                            self.query_one('#baselines', SelectionList).select(baseline)
                        for entry in entries:
                            self.query_one('#datasets', SelectionList).select(entry['dataset'])
                        self.show_dataset(entries[0]['dataset'])
                    for widget in self.query('#experiments OptionList, #experiments DataTable, #experiments .selection-all'):
                        widget.disabled = not editable
                    self.preview()
                else:
                    document = self.document()
                    expand_batch(document, read_config(self.config_path))
                    # Preserve comments and extra batch metadata when updating a file.
                    def save(accepted):
                        if accepted:
                            try:
                                data = read_config(path) if path.exists() else {}
                                data['batch_experiments'] = document['batch_experiments']
                                write_config(path, data)
                                self.notify(f'Saved {path.name}')
                            except Exception as error:
                                self.notify(str(error), severity='error')
                    if path.exists():
                        self.push_screen(Confirm(f'Replace experiment definitions in {path}?', 'Save'), save)
                    else:
                        save(True)
            except Exception as error:
                self.notify(str(error), severity='error')
        elif name == 'new-batch':
            self.imported_batch = None
            self.sequence_choices = {}
            self.query_one('#baselines', SelectionList).deselect_all()
            self.query_one('#datasets', SelectionList).deselect_all()
            if self.active_dataset:
                self.show_dataset(self.active_dataset)
            for widget in self.query('#experiments OptionList, #experiments DataTable, #experiments .selection-all'):
                widget.disabled = False
            self.preview()


    # ------------------------------------------------------------------ Live section

    def live_window_ms(self):
        windows = read_config(self.config_path).get('timewindows') or [50]
        return float(windows[0])

    def live_config(self):
        return read_config(self.config_path)

    def refresh_references(self, select=None):
        listing = self.query_one('#live-references', OptionList)
        names = recordings(self.live_config())
        listing.clear_options()
        listing.add_options(names)
        if names:
            listing.highlighted = names.index(select) if select in names else len(names) - 1
        listing.border_title = eyebrow(f'References · {len(names)}')

    def start_camera(self):
        if not self.camera_running:
            self.camera_running = True
            self.camera_stop.clear()
            self.camera_loop()

    def open_camera(self):
        """The camera as event-capture's gapless mode opens it: every window in order, under an
        on-chip rate ceiling. It stays open for the whole session -- opening an EVK4 takes ~4 s -- so
        recordings start and stop at once, appended window by window to an `EventSink`."""
        import eventcv as ecv

        config = self.live_config()
        kwargs = dict(dt_ms=self.live_window_ms(), latest=False)
        if float(config.get('live_max_event_rate') or 0) > 0:
            kwargs['max_event_rate'] = float(config['live_max_event_rate'])
        if config.get('live_adaptive_bias'):
            kwargs['adaptive_bias'] = True
        try:
            return ecv.stream(**kwargs)
        except Exception:
            if 'max_event_rate' not in kwargs:
                raise
            kwargs.pop('max_event_rate')  # a camera without an on-chip rate controller
            return ecv.stream(**kwargs)

    @on(Input.Changed, '#live-places')
    def places_changed(self, event):
        text = event.value.strip()
        self.live_places = int(text) if text.isdigit() and int(text) > 0 else None

    @work(thread=True, exclusive=True, group='camera')
    def camera_loop(self):
        """Owns the camera: appends to the recording, feeds the localiser, and leaves the newest
        picture in `live_pending` for the UI timer -- it never waits on drawing."""
        import eventcv as ecv

        def status(text):
            self.call_from_thread(self.query_one('#live-camera-status', Static).update, text)

        status('Opening camera…')
        try:
            camera = self.open_camera()
        except Exception as error:
            self.camera_running = False
            status(f'No camera: {error}')
            return
        sink, path, started, drawn = None, None, 0.0, 0.0
        try:
            with camera:
                sensor, name = tuple(camera.sensor_size), f'{camera.name} {camera.serial}'
                window_s = self.live_window_ms() / 1000
                while not self.camera_stop.is_set():
                    request, self.record_request = self.record_request, None
                    if request == 'stop' and sink:
                        events = sink.n_events  # read first: the sink resets its count when it closes
                        sink.finish()
                        self.call_from_thread(self.recording_finished, path, events, sensor, name,
                                              time.monotonic() - started)
                        sink = None
                    elif isinstance(request, Path):
                        try:
                            request.parent.mkdir(parents=True, exist_ok=True)
                            sink, path, started = ecv.EventSink(str(request)), request, time.monotonic()
                        except Exception as error:
                            self.call_from_thread(self.recording_failed, str(error))
                    window = camera.read(timeout_ms=200)
                    if window is None:
                        continue
                    if sink:
                        sink.append(window)
                    localiser = self.localiser
                    feed = localiser is not None and localiser.ready and not localiser.busy.is_set()
                    now = time.monotonic()
                    draw = now - drawn > .045  # windows land every 50 ms, a hair early or late
                    if feed or draw:
                        events = window.numpy()
                        if feed:
                            localiser.send(events, sensor)
                        if draw:
                            line = f'{name} · {len(events) / window_s / 1e6:.2f} Mev/s'
                            if camera.backlog:
                                line += f' · {camera.backlog} behind'
                            if sink:
                                line += f' · ● REC {now - started:.1f} s'
                            drawn, self.live_pending = now, (render(events, sensor), line)
        except Exception as error:
            status(f'Camera stopped: {error}')
        finally:
            if sink:  # the camera stopped under an open recording: keep what was written
                events = sink.n_events
                sink.finish()
                self.call_from_thread(self.recording_finished, path, events, sensor, name, time.monotonic() - started)
            self.camera_running = False

    def show_live_frame(self):
        pending, self.live_pending = self.live_pending, None
        if pending:
            self.live_frame(*pending)

    def live_frame(self, image, line):
        self.query_one('#live-view', PixelView).show(image)
        self.query_one('#live-camera-status', Static).update(line)

    @on(Input.Submitted, '#live-name')
    @on(Button.Pressed, '#live-record')
    def toggle_recording(self):
        button = self.query_one('#live-record', Button)
        if self.recording_name:
            self.record_request = 'stop'
            button.label, button.disabled = 'Stopping…', True
            return
        name = clean_name(self.query_one('#live-name', Input).value)
        if not name:
            self.notify('Name the reference first.', severity='warning')
            self.query_one('#live-name', Input).focus()
            return
        if name in recordings(self.live_config()):
            self.notify(f'A reference called {name} already exists.', severity='warning')
            return
        if not self.camera_running:
            self.notify('The camera is not running.', severity='error')
            return
        self.recording_name = name
        self.record_request = recording_path(self.live_config(), name)
        button.label, button.variant = '■ Stop recording', 'error'

    def recording_finished(self, path, events, sensor, camera, seconds):
        name = self.recording_name or path.stem
        self.recording_name = None
        button = self.query_one('#live-record', Button)
        button.label, button.variant, button.disabled = '● Record', 'primary', False
        if events == 0:
            path.unlink(missing_ok=True)
            self.notify(f'{name}: no events were recorded.', severity='warning')
            return
        register(name, sensor, camera)
        self.datasets[LIVE_DATASET] = recordings(self.live_config())
        self.query_one('#live-name', Input).value = ''
        self.refresh_references(select=name)
        self.notify(f'Saved reference {name}: {events:,} events, {seconds:.1f} s.')

    def recording_failed(self, message):
        self.recording_name = None
        button = self.query_one('#live-record', Button)
        button.label, button.variant, button.disabled = '● Record', 'primary', False
        self.notify(f'Recording failed: {message}', severity='error')

    def selected_reference(self):
        listing = self.query_one('#live-references', OptionList)
        if listing.highlighted is None:
            return None
        return str(listing.get_option_at_index(listing.highlighted).prompt)

    def live_key(self, key):
        """Keys of the Live section: scrubbing a recording, and Delete on the reference list."""
        if self.review is not None:
            per_second = max(1, round(1000 / self.live_window_ms()))
            steps = {'left': -1, 'right': 1, 'shift+left': -per_second, 'shift+right': per_second}
            if key in steps:
                self.show_review(self.review_index + steps[key])
            elif key in ('home', 'end'):
                self.show_review(0 if key == 'home' else self.review.places - 1)
            elif key == 'escape':
                self.close_review()
            else:
                return False
            return True
        if key in ('delete', 'backspace') and self.focused is self.query_one('#live-references'):
            self.delete_reference()
            return True
        return False

    @on(Button.Pressed, '#live-view-ref')
    def toggle_review(self):
        if self.review is not None:
            self.close_review()
            return
        name = self.selected_reference()
        if name is None:
            self.notify('Record or choose a reference first.', severity='warning')
            return
        if self.localiser:
            self.notify('Stop localising to view a recording.', severity='warning')
            return
        self.review_name = name
        self.review = ReferenceView(recording_path(self.live_config(), name), self.live_window_ms())
        self.query_one('#live-view-ref', Button).label = 'Close'
        view = self.query_one('#live-match', PixelView)
        view.can_focus = True  # so the arrow keys reach the viewer rather than the reference list
        view.focus()
        self.show_review(0)

    def show_review(self, index):
        self.review_index = max(0, min(index, self.review.places - 1))
        seconds = self.live_window_ms() / 1000
        self.query_one('#live-match', PixelView).show(self.review.render(self.review_index))
        self.query_one('#live-match-status', Static).update(
            f'{self.review_name} · {self.review_index * seconds:.2f} s / {self.review.places * seconds:.2f} s · '
            f'window {self.review_index + 1}/{self.review.places}\n←→ window · shift ←→ second · home/end · esc close')

    def close_review(self):
        self.review, self.review_name = None, None
        self.query_one('#live-view-ref', Button).label = 'View'
        view = self.query_one('#live-match', PixelView)
        view.can_focus = False
        view.show(None)
        self.query_one('#live-match-status', Static).update('Choose a reference and a method, then Localise.')
        self.query_one('#live-references').focus()

    @on(Button.Pressed, '#live-delete-ref')
    def delete_reference(self):
        name = self.selected_reference()
        if name is None:
            return
        if name == self.recording_name or (self.localiser and self.live_reference == name):
            self.notify(f'{name} is in use; stop recording or localising first.', severity='warning')
            return
        path = recording_path(self.live_config(), name)
        size = path.stat().st_size / 1e6 if path.is_file() else 0

        def confirmed(accepted):
            if not accepted:
                return
            if self.review_name == name:
                self.close_review()
            delete_recording(self.live_config(), name)
            self.datasets[LIVE_DATASET] = recordings(self.live_config())
            self.refresh_references()
            self.notify(f'Deleted reference {name}.')

        self.push_screen(Confirm(f'Delete reference {name}?\n\nThis removes its recording ({size:,.0f} MB) '
                                 'and any cached descriptors.', 'Delete'), confirmed)

    @on(Button.Pressed, '#live-localise')
    def toggle_localising(self):
        if self.localiser:
            self.stop_localising('Stopped.')
            return
        if self.review is not None:
            self.close_review()
        listing = self.query_one('#live-references', OptionList)
        if listing.highlighted is None:
            self.notify('Record or choose a reference first.', severity='warning')
            return
        reference = str(listing.get_option_at_index(listing.highlighted).prompt)
        method = self.query_one('#live-method', Select).value
        path = recording_path(self.live_config(), reference)
        self.live_reference, self.live_place = reference, None
        self.reference_view = ReferenceView(path, self.live_window_ms())
        self.query_one('#live-match', PixelView).show(None)
        self.query_one('#live-match-status', Static).update(f'Mapping {reference} with {method}…')
        self.live_windows = None
        self.localiser = LiveWorker(method, path, self.live_window_ms(), self.live_places,
                                    lambda message: self.call_from_thread(self.localiser_message, message))
        button = self.query_one('#live-localise', Button)
        button.label, button.variant = 'Stop localising', 'error'
        self.start_camera()

    def localiser_message(self, message):
        if self.localiser is None:
            return
        progress = self.query_one('#live-progress', Static)
        status = self.query_one('#live-match-status', Static)
        method = self.localiser.method
        if 'stage' in message:
            progress.update(f"{method} · {message['stage']} {message['done']}/{message['total']}")
        elif 'ready' in message:
            self.live_windows = message['windows']
            progress.update(f"{method} · {message['ready']} places mapped · localising")
        elif 'best' in message:
            best, window = message['best'], message['window']
            places, windows = len(self.live_windows or ()) or 1, self.reference_view.places
            if window != self.live_place and window < windows:  # redrawn only when the place changes
                self.live_place = window
                self.query_one('#live-match', PixelView).show(self.reference_view.render(window))
            place = f'place {best + 1}/{places}' + (f' · window {window + 1}/{windows}' if places < windows else '')
            status.update(f"{self.live_reference} · {place} · score {message['score']:.3f}"
                          f" · {message['ms']:.0f} ms")
        elif 'error' in message:
            if self.localiser.ready:
                status.update(f"{method}: {message['error']}")
            else:
                self.stop_localising(f"{method} failed: {message['error']} (logs/live-{method}.log)")

    def stop_localising(self, message):
        localiser, self.localiser = self.localiser, None
        if localiser:
            threading.Thread(target=localiser.close, daemon=True).start()
        button = self.query_one('#live-localise', Button)
        button.label, button.variant = 'Localise', 'default'
        self.query_one('#live-progress', Static).update(message)

    def on_unmount(self):
        self.camera_stop.set()
        if self.localiser:
            self.localiser.close()


def main():
    os.chdir(ROOT)
    EventLabApp().run()


if __name__ == '__main__':
    main()
