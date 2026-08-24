import sys

class SimpleProgressBar:
    def __init__(self, total, width=40, prefix="Progress",
                 fill="█", faded="▒", empty=" ", end="✓ Done"):
        self.total  = total
        self.width  = width
        self.prefix = prefix
        self.fill   = fill
        self.faded  = faded
        self.empty  = empty
        self.end    = end
        self.count  = 0
        self._last_len = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _bar(self, filled):
        """Build bar string for a given number of filled cells."""
        bar = []
        for i in range(self.width):
            if i < filled - 1:
                bar.append(self.fill)
            elif i == filled - 1:
                bar.append(self.faded)
            else:
                bar.append(self.empty)
        return "".join(bar)

    def _write(self, text):
        """Write text to stdout, padding to overwrite previous line."""
        pad    = max(self._last_len - len(text), 0)
        line   = text + " " * pad
        self._last_len = len(line)
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, step, text=None):
        """Update progress to block `step` (0-indexed)."""
        self.count  = step
        percent     = min(self.count / self.total, 1.0)
        filled      = int(self.width * percent)
        info        = f" | {text}" if text else ""
        self._write(f"{self.prefix}: |{self._bar(filled)}| "
                    f"Block: {self.count}/{self.total}{info}")
        if self.count >= self.total:
            self.finish()

    def show_block(self, block_number):
        """Highlight a single block position on a filled bar."""
        idx = min(int(block_number / self.total * self.width), self.width - 1)
        bar = "".join(self.faded if i == idx else self.fill
                      for i in range(self.width))
        self._write(f"{self.prefix}: |{bar}| Block: {block_number}/{self.total}")

    def finish(self):
        sys.stdout.write(self.end + "\n")
        sys.stdout.flush()