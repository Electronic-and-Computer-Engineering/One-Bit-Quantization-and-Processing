import sys

class SimpleProgressBar:
    def __init__(self, total, width=40, prefix="Progress", fill="█", faded="▒", empty=" ", end="✓ Done"):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.fill = fill
        self.faded = faded
        self.empty = empty
        self.end_msg = end
        self.count = 0
        self._last_line_length = 0

    def update(self, step=1, text=None):
        self.count = step
        percent = min(self.count / self.total, 1.0)
        filled = int(self.width * percent)

        bar = ""
        for i in range(self.width):
            if i < filled - 1:
                bar += self.fill
            elif i == filled - 1:
                bar += self.faded
            else:
                bar += self.empty

        info = f" | {text}" if text else ""
        output = f"\r{self.prefix}: |{bar}| Block: {self.count}/{self.total}{info}"

        # Überschreibe Reste durch Leerzeichen
        pad_length = max(self._last_line_length - len(output), 0)
        output += " " * pad_length
        self._last_line_length = len(output)

        sys.stdout.write(output)
        sys.stdout.flush()

        if self.count >= self.total:
            self.finish()

    def show_block(self, block_number):
        highlight_index = int((block_number / self.total) * self.width)
        highlight_index = min(highlight_index, self.width - 1)

        bar = ""
        for i in range(self.width):
            if i == highlight_index:
                bar += self.faded
            else:
                bar += self.fill

        output = f"\r{self.prefix}: |{bar}| Block: {block_number}/{self.total}"
        pad_length = max(self._last_line_length - len(output), 0)
        output += " " * pad_length
        self._last_line_length = len(output)

        sys.stdout.write(output)
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write(self.end_msg + "\n")
        sys.stdout.flush()