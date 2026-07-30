import os, numpy as np, matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLineEdit, QPushButton

BASE = "TestBatches"

def list_npz():
    out = []
    for root,_,files in os.walk(BASE):
        for f in files:
            if f.endswith(".npz"):
                out.append(os.path.join(root,f))
    return sorted(out)

class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.files = list_npz()
        self.dataX = None
        self.idx = 0
        self.fig = None

        self.cmb = QComboBox(); self.cmb.addItems(self.files)
        self.edt = QLineEdit("0")
        self.prev = QPushButton("<")
        self.next = QPushButton(">")

        lay = QVBoxLayout(self)
        lay.addWidget(self.cmb); lay.addWidget(self.edt)
        lay.addWidget(self.prev); lay.addWidget(self.next)

        self.cmb.currentIndexChanged.connect(self.load)
        self.edt.editingFinished.connect(self.set_idx)
        self.prev.clicked.connect(lambda: self.step(-1))
        self.next.clicked.connect(lambda: self.step(+1))

        if self.files: self.load()

    def load(self):
        d = np.load(self.cmb.currentText())
        self.dataX      = d["mX"]
        self.dataWIdeal = d["mWIdeal"]
        self.dataRIdeal = d["mRIdeal"]
        self.idx = 0
        self.plot()

    def set_idx(self):
        self.idx = max(0, min(int(self.edt.text()), self.dataX.shape[1]-1))
        self.plot()

    def step(self, s):
        self.idx = max(0, min(self.idx+s, self.dataX.shape[1]-1))
        self.edt.setText(str(self.idx))
        self.plot()

    def plot(self):
        if self.fig: plt.close(self.fig)
        self.fig = plt.figure()
        plt.plot(self.dataX[:,self.idx])
        plt.title(f"idx {self.idx}")
        plt.show()

app = QApplication([])
w = Viewer(); w.show()
app.exec()