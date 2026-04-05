import sys

import numpy as np
import aligner_engine.utils as util
from aligner_engine.detector_manager import DetectorManager
from aligner_engine.detector import Detector
import multiprocessing
import threading

import mmcv
import time
import aligner_engine.utils as util
from aligner_engine.version import VERSION
import torch
from aligner_engine import release_util
import os


def hello_world(msg: str):
    text = "DICE Aligner : Hello World! // " + msg
    print(text)
    return text


# This api is for embeddable python in C#, C++
def set_python_executable_path(python_executable_path: str) -> bool:
    DetectorManager.set_python_executable_path(python_executable_path)
    return True


def version():
    return VERSION


# if it succeeds, it will return detector_id, else it will return ""
def load_model(export_path, device="cuda:0") -> str:
    if not release_util.activation_check():
        print('DICE Aligner activation is not confirmed')
        return ""
    if not torch.cuda.is_available():
        device = "cpu"
    detector_id = DetectorManager.load_detector(export_path, device)
    if detector_id != "":
        print("DICE Aligner : Loading Model succeeded! // " + export_path + " on " + device)
    return detector_id


def unload_model(detector_id) -> bool:
    return DetectorManager.unload_detector(detector_id)


def get_rescale_val(detector_id) -> int:
    return DetectorManager.get_rescale_val(detector_id)


def inference(detector_id, img) -> str:
    return DetectorManager.inference(detector_id, img)


def inference_with_rescale(detector_id, img, scale_w_from_api, scale_h_from_api) -> str:
    return DetectorManager.inference(detector_id, img, scale_w_from_api, scale_h_from_api)


def inference_single_bytes(detector_id: str, raw_bytes, w, h, c, ):
    np_array = util.net_bytes_to_numpy(raw_bytes, w, h, c)
    return inference(detector_id, np_array)


def inference_single_bytes_with_rescale(detector_id: str, raw_bytes, w, h, c, scale_w_from_api, scale_h_from_api):
    np_array = util.net_bytes_to_numpy(raw_bytes, w, h, c)
    return inference_with_rescale(detector_id, np_array, scale_w_from_api, scale_h_from_api)

def inference_single_np_with_rescale(detector_id: str, np_array, scale_w_from_api, scale_h_from_api):
    return inference_with_rescale(detector_id, np_array, scale_w_from_api, scale_h_from_api)


def mutilthread_test(idx):
    filename = "D:\\______alinger\\image.jpg"
    # filename_2 = "D:\\__exported\\1.jpg"
    img = mmcv.imread(filename)
    # img_2 = mmcv.imread(filename_2)

    detector_id = load_model("D:\\______alinger\\exported", "cpu")
    print(detector_id)

    for i in range(20):
        start = time.time()
        result = inference(detector_id, img)
        end = time.time()
        print(result)
        print(idx, "--", end - start)

    # print("----------------------------------------------")
    # for i in range(10):
    #     start = time.time()
    #     result = inference(detector_id, img_2)
    #     end = time.time()
    #     # print(result)
    #     print(idx ,"--" , end - start)
    #
    # result = unload_model(detector_id)
    # print(result)
    # detector_id = load_model("D:\\__exported")
    # print(detector_id)
    #
    # print("----------------------------------------------")
    # for i in range(10):
    #     start = time.time()
    #     result = inference(detector_id, img_2)
    #     end = time.time()
    #     # print(result)
    #     print(idx ,"--" , end - start)
    #
    # print("----------------------------------------------")
    # for i in range(10):
    #     start = time.time()
    #     result = inference(detector_id, img)
    #     end = time.time()
    #     # print(result)
    #     print(idx ,"--" , end - start)


if __name__ == "__main__":
    print("test start")

    ######################################################
    # test in single thread

    # filename = "D:\\__exported\\UpperFront_20240111165928.bmp"
    # filename_2 = "D:\\__exported\\1.jpg"
    # img =mmcv.imread(filename)
    # img_2 =mmcv.imread(filename_2)
    #
    # detector_id = load_model("D:\\__exported")
    # print(detector_id)
    #
    # for i in range(10):
    #     start = time.time()
    #     result = inference(detector_id, img)
    #     end = time.time()
    #     # print(result)
    #     print("main" ,"--" , end - start)
    #
    # print("----------------------------------------------")
    # for i in range(10):
    #     start = time.time()
    #     result = inference(detector_id, img_2)
    #     end = time.time()
    #     # print(result)
    #     print("main" ,"--" , end - start)
    #
    # print("----------------------------------------------")
    # for i in range(10):
    #     start = time.time()
    #     result = inference(detector_id, img)
    #     end = time.time()
    #     # print(result)
    #     print("main" ,"--" , end - start)
    #
    # print("----------------------------------------------")
    # for i in range(10):
    #     start = time.time()
    #     result = inference(detector_id, img_2)
    #     end = time.time()
    #     # print(result)
    #     print("main" ,"--" , end - start)

    #################################################################
    # test in multi thread

    p1 = threading.Thread(target=mutilthread_test, args=(1,))
    p2 = threading.Thread(target=mutilthread_test, args=(2,))
    # p3 = threading.Thread(target=mutilthread_test, args=(3,))
    # p4 = threading.Thread(target=mutilthread_test, args=(4,))
    # p5 = threading.Thread(target=mutilthread_test, args=(5,))

    p1.start()
    p2.start()
    # p3.start()
    # p4.start()
    # p5.start()

    p1.join()
    p2.join()
    # p3.join()
    # p4.join()
    # p5.join()

    ##########################################################
    # test with direct detector
    # detector = Detector("D:\\______alinger\\exported\\model_configs.py",
    #                     "D:\\______alinger\\exported\\model.pth",
    #                     "cpu",
    #                     "D:\\______alinger\\exported\\deploy_configs.py",
    #                     "D:\\______alinger\\exported\\end2end.xml",
    #                     enable_vino=True)
    # filename = "D:\\______alinger\\image.jpg"
    # img =mmcv.imread(filename)
    #
    # for i in range(10):
    #     start = time.time()
    #     result=detector.inference(img)
    #     end = time.time()
    #     print(util.detector_result_to_json(result))
    #     print(end - start)

    ##########################################################
    # model load, unload test

    # for i in range(10):
    #     detector_id = load_model("D:\\__exported")
    #     print(detector_id)
    #     result = unload_model(detector_id)
    #     result = unload_model(detector_id)
    #     print(result)

    print("-----------Finished---------------")
