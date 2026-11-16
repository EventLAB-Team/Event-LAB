"""Turn a subprocess stream into log lines and one replaceable progress line."""
import re

ANSI_CONTROL = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def clean_line(text):
    text = ANSI_CONTROL.sub('', text)
    text = re.sub(r'\x1b\[[0-?]*[ -/]*$', '', text)
    return text.replace('\x1b', '').rstrip()


class TerminalOutput:
    def __init__(self):
        self.line = ''
        self.overwrite = False
        self.is_progress = False
        self.progress = ''

    def feed(self, text):
        lines = []
        for char in text:
            if char == '\r':
                self.overwrite = True
            elif char == '\n':
                line = clean_line(self.line)
                if self.is_progress:
                    self.progress = line
                else:
                    if self.progress:
                        lines.append(self.progress)
                        self.progress = ''
                    lines.append(line)
                self.line = ''
                self.overwrite = False
                self.is_progress = False
            else:
                if self.overwrite:
                    self.line = ''
                    self.overwrite = False
                    self.is_progress = True
                self.line += char
        if self.is_progress:
            self.progress = clean_line(self.line)
        return lines

    def finish(self):
        lines = []
        if self.progress:
            lines.append(self.progress)
        if self.line and not self.is_progress and clean_line(self.line):
            lines.append(clean_line(self.line))
        self.__init__()
        return lines
