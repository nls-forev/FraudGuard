import os
import tarfile

import dill
import numpy as np
import yaml

from src.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise e


def convert_to_tar(src_file_path: str, dest_file_path: str):
    try:
        with tarfile.open(dest_file_path, "w:gz") as tar:
            tar.add(src_file_path, arcname=os.path.basename(src_file_path))

        logging.info(f"Successfully converted to tar file at: {dest_file_path}")

    except Exception as e:
        raise e


def load_object(file_path: str) -> object:
    """
    Returns model/object from project directory.

    file_path: str location of file to load

    return: Loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        return obj

    except Exception as e:
        raise e


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save numpy array data to file.

    file_path: str location of file to save
    array: np.ndarray data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array data from file.

    file_path: str location of file to load

    return: np.ndarray
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise e


def load_train_test_data(
    x_train_path: str,
    y_train_path: str,
    x_test_path: str,
    y_test_path: str,
):
    """
    Load transformed train and test datasets.

    Returns
    -------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    """
    try:
        X_train = load_numpy_array_data(x_train_path)
        y_train = load_numpy_array_data(y_train_path)
        X_test = load_numpy_array_data(x_test_path)
        y_test = load_numpy_array_data(y_test_path)

        return X_train, y_train, X_test, y_test

    except Exception as e:
        raise e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils.")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils.")

    except Exception as e:
        raise e
