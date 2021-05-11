r""" A helper module for Practical HW of neural networks, AI course, Sharif University of Technology

Usage:
>>> import hw4_helper
>>> x_train, y_train = hw4_helper.get_train_data()
>>> x_test = hw4_helper.get_test_data()
>>> prediction = np.zeros(100)
>>> hw4_helper.export_prediction(prediction)
>>> hw4_helper.test(prediction)

Author:
    Ahmad Salimi - https://github.com/ahmadsalimi
"""

from operator import truth
from typing import Tuple, List, Iterable
import os
import json
from enum import Enum

import requests
import numpy as np
from tqdm import tqdm


class _FileInfo:

    def __init__(self, name: str, parts: List[str] = ['']):
        self.name = name
        self.parts = parts


CACHE_ROOT = 'data_cache'

DATASET_BASE_URL = 'https://sut-ai.github.io/hw4-base/assets/'
TEST_URL = 'http://sutai.pythonanywhere.com/tester/test/'


class _DataSpecification(Enum):
    TrainX = _FileInfo('x_train.npy', ['.aa', '.ab', '.ac'])
    TrainY = _FileInfo('y_train.npy')
    TestX = _FileInfo('x_test.npy')


def _download_file(responses: Iterable[requests.Response], size: int, filename: str):
    CHUNK_SIZE = 32768
    with open(filename, 'wb') as file:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=size) as bar:
            for response in responses:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)
                        bar.update(len(chunk))


def _get_response(filename: str, part: str) -> requests.Response:
    return requests.get(f'{DATASET_BASE_URL}{filename}{part}', stream=True)


def _get_size(file_info: _FileInfo) -> int:
    responses = map(lambda part: _get_response(
        file_info.name, part), file_info.parts)
    return sum(map(lambda res: int(res.headers['Content-Length']), responses))


def _get_data_by_file_info(file_info: _FileInfo) -> np.ndarray:
    responses = map(lambda part: _get_response(
        file_info.name, part), file_info.parts)
    os.makedirs(CACHE_ROOT, exist_ok=True)
    _download_file(responses, _get_size(file_info),
                   os.path.join(CACHE_ROOT, file_info.name))


def _get_data(file_info: _FileInfo) -> np.ndarray:
    file_path = os.path.join(CACHE_ROOT, file_info.name)
    if not os.path.exists(file_path):
        _get_data_by_file_info(file_info)

    return np.load(file_path)


def get_train_data() -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Downloads the train dataset (if needed) and returns

            Returns:
                    x (np.ndarray): images of the dataset.  shape: (N, 28, 28)
                    y (np.ndarray): labels of the images.   shape: (N,)
    """
    x = _get_data(_DataSpecification.TrainX.value)
    y = _get_data(_DataSpecification.TrainY.value)
    return x, y


def get_test_data() -> np.ndarray:
    r"""
    Downloads the test dataset (if needed) and returns

            Returns:
                    x (np.ndarray): images of the dataset.  shape: (N, 28, 28)
    """
    return _get_data(_DataSpecification.TestX.value)


def export_prediction(prediction: np.ndarray):
    r"""
    Exports the given prediction array to `prediction.npy`

            Parameters:
                    prediction (np.ndarray): The prediction array. (one dimensional)
    """
    assert isinstance(prediction, np.ndarray),\
        f"Expected prediction to be a numpy.ndarray, got {type(prediction)}"
    assert prediction.ndim == 1,\
        f"Expected prediction to be a 1D array, got a {prediction.ndim} dimensional array"
    y_test = get_test_data()
    assert y_test.shape[0] == prediction.shape[0],\
        f"Expected prediction shape to be {(y_test.shape[0],)}, Got {prediction.shape}"
    np.save('prediction', prediction.astype(np.uint8))


def test(prediction: np.ndarray):
    r"""
    Tests the given prediction array using test API

            Parameters:
                    prediction (np.ndarray): The prediction array. (one dimensional)
    """
    assert isinstance(prediction, np.ndarray),\
        f"Expected prediction to be a numpy.ndarray, got {type(prediction)}"
    assert prediction.ndim == 1,\
        f"Expected prediction to be a 1D array, got a {prediction.ndim} dimensional array"
    y_test = get_test_data()
    assert y_test.shape[0] == prediction.shape[0],\
        f"Expected prediction shape to be {(y_test.shape[0],)}, Got {prediction.shape}"

    res = requests.post(TEST_URL, data=prediction.astype(np.uint8).tobytes())

    if res.status_code == 500:
        print('Internal server error! Contact Ahmad Salimi :))))')
        return

    try:
        message = json.loads(res.content)['message']
        print(f'Status: {res.status_code} - message: {message}')
    except:
        print('Unable to parse the response! Contact Ahmad Salimi :))))')
