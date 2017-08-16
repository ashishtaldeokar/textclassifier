from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

class word_frequency():
    """
    Plot word frequencies
    """
    def __init__(self, wl):
        self.word_list = wl
        self.counts = Counter(self.word_list)
        self.labels, self.values = zip(*self.counts.items())
        self.indSort = np.argsort(self.values)[::-1]
        self.labels = np.array(self.labels)[self.indSort]
        self.values = np.array(self.values)[self.indSort]
        self.indexes = np.arange(len(self.labels))
        self.bar_width = 0.3
        self.plt = plt

    def plot(self):
        self.plt.bar(self.indexes, self.values)
        self.plt.xticks(self.indexes + self.bar_width, self.labels)
        self.plt.show()
