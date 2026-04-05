import yaml
import os
import glob
import os.path as osp
import json
import numpy as np
import ctypes
import shutil
from shutil import copyfile

import clr   #for pythonnet
from System.Runtime.InteropServices import GCHandle, GCHandleType #for pythonnet



ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


def join_path(*paths):
    osp_path = osp.join(*paths)
    return osp_path.replace('\\', '/')

def clear_dir(path_name):
    try:
        shutil.rmtree(path_name)
    except:
        pass

def is_exist(path):
    return os.path.exists(path)

def remove_file(path):
    if is_exist(path):
        try:
            os.remove(path)
            return True
        except:
            pass
    return False

def write_yaml(file_name, data):
    make_dir(file_name)
    with open(file_name, 'w') as yaml_file:
        yaml.safe_dump(data, yaml_file, default_flow_style=False)

def read_yaml(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def copy_file(src, dst):

    try:
        copyfile(src, dst)
        return True
    except:
        pass
    return False


def make_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def read_json(file):
    with open(file, encoding='utf-8') as j:
        data = json.load(j)
    return data

def save_json(object_in_json, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(object_in_json, f, ensure_ascii=False, indent=4)


def detector_result_to_json(inference_result: dict):
    json_result = {}
    for idx, data in inference_result.items():
        sub_result = {}
        sub_result["class_name"] = data["class_name"]
        sub_result["conf"] = data["conf"]
        sub_result["qbox"] = data["qbox"]
        sub_result["center"] = data["center"]
        sub_result["longside"] = data["longside"]
        sub_result["shortside"] = data["shortside"]
        sub_result["angle_degree"] = data["angle_degree"]
        sub_result["angle_radian"] = data["angle_radian"]

        json_result[idx] = sub_result
    return json.dumps(json_result)



_MAP_NET_NP = {
    'Single': np.dtype('float32'),
    'Double': np.dtype('float64'),
    'SByte': np.dtype('int8'),
    'Int16': np.dtype('int16'),
    'Int32': np.dtype('int32'),
    'Int64': np.dtype('int64'),
    'Byte': np.dtype('uint8'),
    'UInt16': np.dtype('uint16'),
    'UInt32': np.dtype('uint32'),
    'UInt64': np.dtype('uint64'),
    'Boolean': np.dtype('bool'),
}

def asNumpyArray(netArray):
    '''
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for
    the mapping of CLR types to Numpy dtypes.
    '''
    dims = np.empty(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)
    netType = netArray.GetType().GetElementType().Name
    try:
        npArray = np.empty(dims, order='C', dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType))
    try:  # Memmove
        sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        destPtr = npArray.__array_interface__['data'][0]
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if sourceHandle.IsAllocated: sourceHandle.Free()
    return npArray


def net_bytes_to_numpy(bytes, w, h, c):
    npArray = asNumpyArray(bytes)
    npArray = npArray.reshape((h, w, c), order='C')
    return npArray
