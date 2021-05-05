import os
from unittest import TestCase
import numpy as np

TEST = 'y_test.npy'
PRED = 'prediction.npy'


class Test(TestCase):

    @classmethod
    def setUpClass(cls):
        "Hook method for setting up class fixture before running tests in the class."

        assert os.path.exists(TEST), "Test file does not exist!"
        y_test = np.load(TEST)

        assert os.path.exists(PRED), "Prediction file does not exist!"
        y_pred = np.load(PRED)

        assert y_test.shape == y_pred.shape,\
            f"Expected prediction shape to be {y_test.shape}, Got {y_pred.shape}"

        accuracy = (y_test == y_pred).astype(float).mean() * 100

        print(f'The accuracy of your submission is {accuracy:.2f}')

        for i in range(100):
            setattr(cls, f'accuracy_g{i}', lambda self: self.assertGreater(accuracy, i))
