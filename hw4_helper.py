from typing import Tuple
import os
from enum import Enum
from collections import namedtuple
import requests
from http.cookies import SimpleCookie
import io
import numpy as np
from tqdm import tqdm

FileInfo = namedtuple('FileInfo', 'id name')
CACHE_ROOT = 'data_cache'

class DataSpecification(Enum):
    TrainX = FileInfo('1-0iZwp7vygQqXNLaDmORp_d_EDQVrBXz', os.path.join(CACHE_ROOT, 'x_train.npy'))
    TrainY = FileInfo('1-4be9NCtS_fhFePJP1T_92iAhwCvwGuQ', os.path.join(CACHE_ROOT, 'y_train.npy'))
    TestX = FileInfo('1-4A5ZY2jdOFKnupZ2eBB2P4EJI-4im7p', os.path.join(CACHE_ROOT, 'x_test.npy'))

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def _get_gdrive_response(file_id: str) -> requests.Response:
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    cookie = SimpleCookie()
    cookie.load(response.headers['Set-Cookie'])

    session.cookies.update(cookie)
    response.cookies.update(cookie)
    token = get_confirm_token(response)

    print('token:', token, response.cookies.keys(), response.headers.keys())

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        print('here')

    return response


def _download_file(response: requests.Response, filename: str):
    print(response)
    CHUNK_SIZE = 32768
    with open(filename, 'wb') as file:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
                    bar.update(CHUNK_SIZE)


def _get_data_by_file_info(file_info: FileInfo) -> np.ndarray:
    response = _get_gdrive_response(file_info.id)
    os.makedirs(CACHE_ROOT, exist_ok=True)
    _download_file(response, file_info.name)


def _get_data(file_info: FileInfo) -> np.ndarray:
    if not os.path.exists(file_info.name):
        _get_data_by_file_info(file_info)

    return np.load(file_info.name)


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
