import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

class plot():
    """
    Class to plot charts like confusion matrix
    """
    def __init__(self):

        self.plt = plt
        self.CM = confusion_matrix

    def cnf_mtx(self, truth, prediction, name=" No Name given "):
        """
        Plot a confusion matrix
        """

        mtx = self.CM(truth, prediction)
        self.plt.imshow(mtx, interpolation='nearest', cmap=plt.cm.Blues)
        self.plt.title("Confusion Matrix")
        self.plt.colorbar()
        self.plt.title(name)
        self.plt.ylabel(" Ground Truth ")
        self.plt.xlabel(" Classifier Prediction ")
        self.plt.show()

