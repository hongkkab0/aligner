from aligner_engine.detector import Detector
from typing import List, Set, Dict, Tuple
import aligner_engine.utils as util
import aligner_engine.const as const
import numpy as np
import traceback
import multiprocessing
from multiprocessing import Process, Pipe, shared_memory
from multiprocessing.connection import PipeConnection
import os, sys
import threading
from mmengine.config import Config
import mmcv
import cv2
import math


class TopicType:
    TOPIC_ERROR = "TOPIC_ERROR"
    TOPIC_LOAD_DETECTOR_SUCCESS = "TOPIC_LOAD_DETECTOR_SUCCESS"
    TOPIC_UNLOAD_DETECTOR = "TOPIC_UNLOAD_DETECTOR"
    TOPIC_UNLOAD_DETECTOR_SUCCESS = "TOPIC_UNLOAD_DETECTOR_SUCCESS"
    TOPIC_INFERENCE = "TOPIC_INFERENCE"
    TOPIC_INFERENCE_RESULT = "TOPIC_INFERENCE_RESULT"


# This loop is executed in another process.
def bootstrap_detector(pipe_to_detector_rcv: PipeConnection,
                       pipe_from_detector_send: PipeConnection,
                       export_path: str, device: str, detector_id: str):
    message_out = {"topic": "", "payload": {}}
    detector: Detector = None
    try:
        config_path = util.join_path(export_path, const.FILENAME_MODEL_CONFIG)
        ckpt_path = util.join_path(export_path, const.FILENAME_CKPT)
        deploy_config_path = util.join_path(export_path, const.FILENAME_DEPLOY_CONFIG)
        vino_xml_path = util.join_path(export_path, const.FILENAME_EXPORT_VINO_XML)
        detector = Detector(config_path, ckpt_path, device, deploy_config_path, vino_xml_path)

        cfg = Config.fromfile(config_path)
        test_pipeline = cfg.test_dataloader.dataset.pipeline
        rescale_value = (0, 0)
        for idx, transform in enumerate(test_pipeline):
            if transform.type == "mmdet.Resize":
                rescale_value = transform.scale

        empty_img = np.zeros((rescale_value[0], rescale_value[1], 3), dtype=np.uint8)
        detector.inference(empty_img)  # to alleviate slow initial inference
        message_out["topic"] = TopicType.TOPIC_LOAD_DETECTOR_SUCCESS
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        message_out["topic"] = TopicType.TOPIC_ERROR
        message_out["payload"] = {"detail": str(e)}

    finally:
        pipe_from_detector_send.send(message_out)

    if message_out["topic"] == TopicType.TOPIC_ERROR:
        return

    is_run = True
    while is_run:
        message_in = pipe_to_detector_rcv.recv()
        message_out = {"topic": "", "payload": {}}

        try:
            if message_in["topic"] == TopicType.TOPIC_UNLOAD_DETECTOR:
                is_run = False
                detector.unload_model()
                message_out["topic"] = TopicType.TOPIC_UNLOAD_DETECTOR_SUCCESS
                message_out["payload"] = {}
            elif message_in["topic"] == TopicType.TOPIC_INFERENCE:
                shm = shared_memory.SharedMemory(name=detector_id)
                img_h = message_in["payload"]["img_h"]
                img_w = message_in["payload"]["img_w"]
                img_c = message_in["payload"]["img_c"]

                arr = np.frombuffer(buffer=shm.buf[:img_h * img_w * img_c], dtype=np.uint8)
                arr = arr.reshape((img_h, img_w, img_c))
                result = detector.inference(arr)
                message_out["topic"] = TopicType.TOPIC_INFERENCE_RESULT
                message_out["payload"] = result
                del arr
                shm.close()

        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(e)
            message_out["topic"] = TopicType.TOPIC_ERROR
            message_out["payload"] = {"detail": str(e)}
        finally:
            pipe_from_detector_send.send(message_out)


class DetectorManager:
    _pipe_to_detector_send: Dict[str, PipeConnection] = {}  # pairs of detector_id and Pipe in
    _pipe_from_detector_rcv: Dict[str, PipeConnection] = {}  # pairs of detector_id and Pipe out
    _detector_id_accum = 0  # it will be accumulated when a model is loaded.
    _shared_memory_buffer_sizes: Dict[str, int] = {}
    _shared_memories: Dict[str, shared_memory.SharedMemory] = {}  # share
    _rescale_values = {}
    _lock = threading.Lock()
    _python_executable_path = ""

    @classmethod
    def load_detector(cls, export_path, device) -> str:
        config_path = util.join_path(export_path, const.FILENAME_MODEL_CONFIG)

        detector_id = cls._get_detector_id()
        pipe_to_detector_rcv, pipe_to_detector_send = Pipe(duplex=False)
        pipe_from_detector_rcv, pipe_from_detector_send = Pipe(duplex=False)

        if cls._python_executable_path != "":
            multiprocessing.set_executable(cls._python_executable_path)
        process = Process(target=bootstrap_detector,
                          args=(pipe_to_detector_rcv, pipe_from_detector_send,
                                export_path, device, detector_id))
        process.daemon = True
        process.start()

        # check if the model is loaded successfully.
        message_out = pipe_from_detector_rcv.recv()
        if message_out["topic"] == TopicType.TOPIC_LOAD_DETECTOR_SUCCESS:
            cls._pipe_to_detector_send[detector_id] = pipe_to_detector_send
            cls._pipe_from_detector_rcv[detector_id] = pipe_from_detector_rcv

            cfg = Config.fromfile(config_path)
            test_pipeline = cfg.test_dataloader.dataset.pipeline
            rescale_value = (0, 0)
            for idx, transform in enumerate(test_pipeline):
                if transform.type == "mmdet.Resize":
                    rescale_value = transform.scale

            cls._rescale_values[detector_id] = rescale_value
            cls._shared_memory_buffer_sizes[detector_id] = rescale_value[0] * rescale_value[1] * 3
            shm = shared_memory.SharedMemory(create=True,
                                             size=cls._shared_memory_buffer_sizes[detector_id],
                                             name=detector_id)
            cls._shared_memories[detector_id] = shm

            return detector_id
        else:
            print(message_out["payload"])
            return ""

    @classmethod
    def unload_detector(cls, detector_id: str) -> bool:
        try:
            pipe_to_detector_send = cls._pipe_to_detector_send[detector_id]
            pipe_from_detector_rcv = cls._pipe_from_detector_rcv[detector_id]

            pipe_to_detector_send.send({"topic": TopicType.TOPIC_UNLOAD_DETECTOR, "payload": {}})
            message_out = pipe_from_detector_rcv.recv()
            if message_out["topic"] == TopicType.TOPIC_ERROR:
                return False
            else:
                cls._pipe_to_detector_send.pop(detector_id)
                cls._pipe_from_detector_rcv.pop(detector_id)
                if cls._shared_memory_buffer_sizes[detector_id] > 0:
                    shm = cls._shared_memories[detector_id]
                    shm.close()
                    shm.unlink()
                    cls._shared_memories.pop(detector_id)

                cls._shared_memory_buffer_sizes.pop(detector_id)
                return True
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
            return False

    @classmethod
    def get_rescale_val(cls, detector_id: str) -> int:
        try:
            return cls._rescale_values[detector_id][0]
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
            return 0

    @classmethod
    def inference(cls, detector_id: str, img: np.ndarray, scale_w_from_api=1.0, scale_h_from_api=1.0) -> str:
        try:
            pipe_to_detector_send = cls._pipe_to_detector_send[detector_id]
            pipe_from_detector_rcv = cls._pipe_from_detector_rcv[detector_id]

            h_original, w_original = img.shape[:2]
            w_scale = 1.0
            h_scale = 1.0
            if h_original > cls._rescale_values[detector_id][0] or w_original > cls._rescale_values[detector_id][0]:
                img, scale_factor = mmcv.imrescale(
                    img,
                    cls._rescale_values[detector_id],
                    interpolation="bilinear",
                    return_scale=True,
                    backend="cv2")

                h_new, w_new = img.shape[:2]
                w_scale = w_new / w_original
                h_scale = h_new / h_original

            if img.nbytes > cls._shared_memory_buffer_sizes[
                detector_id]:  # enlarge shared memory for inference
                shm = cls._shared_memories[detector_id]
                shm.close()
                shm.unlink()
                shm = shared_memory.SharedMemory(create=True, size=img.nbytes, name=detector_id)
                cls._shared_memories[detector_id] = shm
                cls._shared_memory_buffer_sizes[detector_id] = img.nbytes
            else:
                shm = cls._shared_memories[detector_id]

            shared_img = np.frombuffer(shm.buf, img.dtype)

            img_h = img.shape[0]
            img_w = img.shape[1]
            img_c = img.shape[2]
            shared_img[:img_h * img_w * img_c] = img.flatten()

            pipe_to_detector_send.send({
                "topic": TopicType.TOPIC_INFERENCE,
                "payload": {
                    "img_h": img_h, "img_w": img_w, "img_c": img_c
                }
            })

            message_out = pipe_from_detector_rcv.recv()
            message_out_payload = message_out["payload"]

            for box_id, box in message_out_payload.items():
                qbox = box['qbox']  # [x1 y1 x2 y2 x3 y3 x4 y4]
                qbox_scaled = [qbox[0]/w_scale, qbox[1]/h_scale,
                               qbox[2]/w_scale, qbox[3]/h_scale,
                               qbox[4]/w_scale, qbox[5]/h_scale,
                               qbox[6]/w_scale, qbox[7]/h_scale]
                box['qbox'] = qbox_scaled

            if (scale_w_from_api != 1.0) or (scale_h_from_api != 1.0):
                for box_id, box in message_out_payload.items():
                    qbox = box['qbox']
                    qbox_scaled = [qbox[0] / scale_w_from_api, qbox[1] / scale_h_from_api,
                                   qbox[2] / scale_w_from_api, qbox[3] / scale_h_from_api,
                                   qbox[4] / scale_w_from_api, qbox[5] / scale_h_from_api,
                                   qbox[6] / scale_w_from_api, qbox[7] / scale_h_from_api]
                    box['qbox'] = qbox_scaled

            for box_id, box in message_out_payload.items():
                qbox = box['qbox']
                (cx, cy), (w, h), angle = cv2.minAreaRect(
                    np.array([[qbox[0], qbox[1]],
                              [qbox[2], qbox[3]],
                              [qbox[4], qbox[5]],
                              [qbox[6], qbox[7]]], dtype=np.int32))
                box['center'] = [cx, cy]
                box['longside'] = max(w, h)
                box['shortside'] = min(w, h)
                box['angle_degree'] = angle
                box['angle_radian'] = angle * math.pi / 180

            return_str = util.detector_result_to_json(message_out_payload)
            return return_str
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
            return ""

    @classmethod
    def _get_detector_id(cls) -> str:
        cls._lock.acquire()
        detector_id = "__aligner-" + str(cls._detector_id_accum) + "__"
        cls._detector_id_accum = cls._detector_id_accum + 1
        cls._lock.release()
        return detector_id

    @classmethod
    def set_python_executable_path(cls, python_executable_path):
        cls._python_executable_path = python_executable_path
