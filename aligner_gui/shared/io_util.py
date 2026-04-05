import json
import logging
import os
import traceback
from datetime import datetime

import socket
import yaml
# import psutil

from pathlib import Path

def join_path(*path_list):
    return os.path.join(*path_list)


def get_user_home_dir():
    return str(Path.home())


def get_aligner_home_dir():
    return join_path(get_user_home_dir(), '.dice_aligner')

def make_dir(path):
    dir_path = os.path.dirname(path)
    if not is_exist(dir_path):
        os.makedirs(dir_path)


def is_exist(path):
    return os.path.exists(path)


def is_dir(path):
    return os.path.isdir(path)


def remove_file(path):
    if is_exist(path):
        try:
            os.remove(path)
            return True
        except:
            pass
    return False


def read(path):
    data = ''
    try:
        with open(path, 'r', encoding='UTF8') as file:
            data = file.read()
    except Exception as e:
        print(e)
    return data


def read_lines(path):
    lines = []
    try:
        with open(path, 'r', encoding='UTF8') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip()
    except Exception as e:
        print(e)
    return lines


def write(path, text):
    remove_file(path)
    try:
        with open(path, 'w', encoding='UTF8', newline='') as file:
            file.write(str(text))
    except Exception as e:
        print(e)


def write_lines(path, lines):
    remove_file(path)
    try:
        with open(path, 'w', encoding='UTF8', newline='') as file:
            for line in lines:
                file.write(str(line) + os.linesep)
    except Exception as e:
        print(e)


def dump_yaml(path: str, data: dict):
    make_dir(path)
    with open(path, 'w', encoding='UTF8') as outfile:
        yaml.safe_dump(data, outfile, default_flow_style=False)
        print('dump yaml - ', path)


def load_yaml(path: str, default_data=None):
    data = {}
    try:
        with open(path, 'r', encoding='UTF8') as readfile:
            data = yaml.safe_load(readfile)
            print('load yaml - ', path)
    except:
        pass

    if default_data is not None:
        for key in default_data.keys():
            if key in data.keys():
                default_data[key] = data[key]
        return default_data

    return data


def get_file_list(path):
    file_list = os.listdir(path)
    all_files = list()
    for entry in file_list:
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_file_list(full_path)
        else:
            all_files.append(full_path)

    return all_files


def read_json(file_name):
    with open(file_name, encoding='UTF8') as j:
        data = json.load(j)
    return data

