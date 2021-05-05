import os
from unittest import TestCase
import numpy as np

TEST = 'y_test.npy'
PRED = 'prediction.npy'


class Test(TestCase):

    def test(self):
        self.assertTrue(os.path.exists(TEST), "Test file does not exist!")
        y_test = np.load(TEST)

        self.assertTrue(os.path.exists(PRED),
                        "Prediction file does not exist!")
        y_pred = np.load(PRED)

        self.assertEqual(y_test.shape, y_pred.shape,
                         f"Expected prediction shape to be {y_test.shape}, Got {y_pred.shape}")

        accuracy = (y_test == y_pred).astype(float).mean() * 100

        print(f'The accuracy of your submission is {accuracy:.2f}')
