import logging
import os
import traceback
import uuid
from pathlib import Path
import sys
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
import aligner_engine.utils as util
import psutil
from aligner_gui.utils import io_util
import wmi

def get_mac_addrs():
    try:
        mac_addrs = []
        net_infos = psutil.net_if_addrs()
        for interface_name, interface_infos in net_infos.items():
            for interface_info in interface_infos:
                if interface_info.family == psutil.AF_LINK:
                    mac_addrs.append(interface_info.address)
                    break
        # mac_addrs.sort()
        # print(mac_addrs)
        if len(mac_addrs) == 0:
            return [""]
        return mac_addrs
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
        return [""]


def get_disk_id():
    try:
        c = wmi.WMI()

        os_drive = c.Win32_OperatingSystem()[0].SystemDrive

        for logical_disk in c.Win32_LogicalDisk(DeviceID=os_drive):
            physical_disk = \
                logical_disk.associators("Win32_LogicalDiskToPartition")[0].associators(
                    "Win32_DiskDriveToDiskPartition")[0]
            volume_serial_number = physical_disk.SerialNumber
            system_drive_uuid = uuid.uuid5(uuid.NAMESPACE_OID, volume_serial_number)
            return str(system_drive_uuid)

    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
        return ""


def get_activation_path():
    path = util.join_path(util.ROOT_PATH, "aligner_activation.key")
    return path


def get_activation_key():
    path = get_activation_path()
    key = None
    if io_util.is_exist(path):
        try:
            with open(path, 'r') as f:
                key = f.readline()
        except Exception as e:
            print(e)
    return key


class SHACipher:
    PREFIX = "DICE-ALIGNER_"
    CODE_LEN= 20
    CODE_LEN_NEW = 5

    @classmethod
    def get_activation_key(cls, device_id):
        device_id_prefix = cls.PREFIX + device_id
        device_id_prefix_encoded = hashlib.sha256(device_id_prefix.encode()).digest()
        device_id_prefix_encoded_b32 = base64.b32encode(device_id_prefix_encoded[:cls.CODE_LEN]).decode('utf-8')
        return device_id_prefix_encoded_b32


    @classmethod
    def activation_check_with_key(cls, device_id, activation_key):
        try:
            device_id_prefix = cls.PREFIX + device_id
            device_id_prefix_encoded = hashlib.sha256(device_id_prefix.encode()).digest()
            device_id_prefix_encoded_b32 = base64.b32encode(device_id_prefix_encoded[:cls.CODE_LEN]).decode('utf-8')
            if device_id_prefix_encoded_b32 == activation_key:
                return True
            new_device_id_prefix_encoded_b32 = base64.b32encode(device_id_prefix_encoded[:cls.CODE_LEN_NEW]).decode('utf-8')
            return  new_device_id_prefix_encoded_b32 == activation_key

        except Exception as e:
            logging.info(e)
            return False


class AESCipher:
    key_str = str(434523576456 // 23 * 10)
    sep = "_"
    prefix = "DICE-ALIGNER"

    def __init__(self, key=key_str):
        self.bs = 32
        self.key = hashlib.sha256(AESCipher.str_to_bytes(key)).digest()

    @staticmethod
    def str_to_bytes(data):
        u_type = type(b''.decode('utf8'))
        if isinstance(data, u_type):
            return data.encode('utf8')
        return data

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * AESCipher.str_to_bytes(
            chr(self.bs - len(s) % self.bs))

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s) - 1:])]

    def encrypt(self, raw):
        raw = self._pad(AESCipher.str_to_bytes(raw))
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw)).decode('utf-8')

    def decrypt(self, enc, is_new=True):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')


def activation_check():
    mac_addrs = get_mac_addrs()
    disk_id = get_disk_id()
    activation_key = get_activation_key()

    activation_methods = [
        lambda: any(activation_check_with_key(mac_addr, activation_key) for mac_addr in mac_addrs),
        lambda: activation_check_with_key(disk_id, activation_key),
        lambda: SHACipher.activation_check_with_key(disk_id, activation_key)
    ]

    return any(method() for method in activation_methods)


def activation_check_with_key(device_id, activation_key):
    cipher = AESCipher()
    try:
        raw = cipher.decrypt(activation_key)
        tokens = raw.split(AESCipher.sep)
        if tokens[0] != AESCipher.prefix:
            return False

        key_id = AESCipher.sep.join(tokens[1:])
        return key_id == device_id

    except ValueError as e:
        logging.info("activation_key is invalid.")
        logging.info("current activation_key : " + activation_key)
        return False





# if the version < 1.7.0
# if __name__ == "__main__":
#     passwoard = sys.argv[1]
#     if passwoard == 'Alinger_admin!':
#         serial_id = sys.argv[2]
#         raw = AESCipher.prefix + AESCipher.sep + serial_id
#         cipher = AESCipher()
#         encrypted = cipher.encrypt(raw)
#         decoded = cipher.decrypt(encrypted)
#         print("serial_id : " + serial_id)
#         print("raw : " + raw)
#         print("encrypted : " + encrypted)
#         print("decoded : " + decoded)
#         print("")
#         print("KEY (아래 줄 카피해주세요)")
#         print(encrypted)
#     else:
#         print("Who are you?")


# if the version >= 1.7.1
# if __name__ == "__main__":
#     passwoard = sys.argv[1]
#     if passwoard == 'Alinger_admin!':
#         serial_id = sys.argv[2]
#
#         encrypted = SHACipher.get_activation_key(serial_id)
#         print("serial_id : " + serial_id)
#         print("encrypted : " + encrypted)
#         print("activation test result : ", SHACipher.activation_check_with_key(serial_id, encrypted))
#         print("")
#         print("KEY (아래 줄 카피해주세요)")
#         print(encrypted)
#     else:
#         print("Who are you?")
