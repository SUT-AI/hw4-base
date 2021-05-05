from typing import Tuple, List, Iterable
import os
from enum import Enum
from collections import namedtuple
import requests
from http.cookies import SimpleCookie
import io
import numpy as np
from tqdm import tqdm


class FileInfo:

    def __init__(self, name: str, parts: List[str] = ['']):
        self.name = name
        self.parts = parts


CACHE_ROOT = 'data_cache'

BASE_URL = 'https://sut-ai.github.io/hw4-base/assets/'


class DataSpecification(Enum):
    TrainX = FileInfo('x_train.npy', ['.aa', '.ab', '.ac'])
    TrainY = FileInfo('y_train.npy')
    TestX = FileInfo('x_test.npy')


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
    return requests.get(f'{BASE_URL}{filename}{part}', stream=True)


def _get_size(file_info: FileInfo) -> int:
    responses = map(lambda part: _get_response(file_info.name, part), file_info.parts)
    return sum(map(lambda res: int(res.headers['Content-Length']), responses))


def _get_data_by_file_info(file_info: FileInfo) -> np.ndarray:
    responses = map(lambda part: _get_response(file_info.name, part), file_info.parts)
    os.makedirs(CACHE_ROOT, exist_ok=True)
    _download_file(responses, _get_size(file_info), os.path.join(CACHE_ROOT, file_info.name))


def _get_data(file_info: FileInfo) -> np.ndarray:
    file_path = os.path.join(CACHE_ROOT, file_info.name)
    if not os.path.exists(file_path):
        _get_data_by_file_info(file_info)

    return np.load(file_path)


def get_train_data() -> Tuple[np.ndarray, np.ndarray]:
    x = _get_data(DataSpecification.TrainX.value)
    y = _get_data(DataSpecification.TrainY.value)
    return x, y


def get_test_data() -> np.ndarray:
    return _get_data(DataSpecification.TestX.value)


def export_prediction(prediction: np.ndarray):
    assert isinstance(prediction, np.ndarray),\
        f"Expected prediction to be a numpy.ndarray, got {type(prediction)}"
    assert prediction.ndim == 1,\
        f"Expected prediction to be a 1D array, got a {prediction.ndim} dimensional array"
    np.save('prediction', prediction.astype(np.uint8))
