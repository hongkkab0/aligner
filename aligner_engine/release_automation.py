import sys
import os
import aligner_engine.utils as util
import subprocess
from distutils.dir_util import copy_tree
import shutil
import tqdm


def scan_all_files(folder_path):
    results = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            relative_path = root.replace("\\", '/') + '/' + file
            results.append(relative_path)
    results.sort(key=lambda x: x.lower())
    return results


def delete_all_pycache(folder_path):
    dir_paths = [x[0] for x in os.walk(folder_path)]
    pycaches = []
    for dir_path in dir_paths:
        if os.path.basename(dir_path) == '__pycache__':
            pycaches.append(dir_path)

    for pycach in pycaches:
        shutil.rmtree(pycach)


def encrypt_with_pyarmor(src, dst):
    args = ['pyarmor', 'gen', '-O', dst, '-r', src]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, err = proc.communicate()


def copy_venv(src_dir, dest_dir):
    src_dir = src_dir.replace("\\", '/')
    dest_dir = dest_dir.replace("\\", '/')
    src_dir_len = len(src_dir)
    src_file_paths = scan_all_files(src_dir)
    for src_file_path in tqdm.tqdm(src_file_paths):
        dst_file_path = dest_dir + src_file_path[src_dir_len:]
        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
        shutil.copy(src_file_path, dst_file_path)


def release(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    files = scan_all_files(target_dir)

    if len(files) > 0:
        raise Exception("Release failed. Because there are files in target directory."
              "\r\nPlease clear target directory before release.")

    print("Encrypting...")
    aligner_encrypted_path = os.path.abspath(os.path.join(target_dir, 'encrypted'))
    encrypt_with_pyarmor(os.path.abspath(os.path.join(util.ROOT_PATH, 'aligner_engine')), aligner_encrypted_path)
    encrypt_with_pyarmor(os.path.abspath(os.path.join(util.ROOT_PATH, 'aligner_gui')), aligner_encrypted_path)

    print("Copying files...")
    copy_tree(os.path.join(util.ROOT_PATH, 'aligner_pretrained'),
              os.path.join(target_dir, 'aligner_pretrained'))
    copy_tree(os.path.join(util.ROOT_PATH, 'aligner_sample'),
              os.path.join(target_dir, 'aligner_sample'))
    copy_tree(os.path.join(util.ROOT_PATH, 'python-3.9.13-embed-amd64'),
              os.path.join(target_dir, 'python-3.9.13-embed-amd64'))
    copy_tree(os.path.join(util.ROOT_PATH, 'aligner_engine'),
              os.path.join(target_dir, 'aligner_engine'))
    copy_tree(os.path.join(util.ROOT_PATH, 'aligner_gui'),
              os.path.join(target_dir, 'aligner_gui'))
    util.copy_file(os.path.join(util.ROOT_PATH, 'dice_aligner.bat'),
                   os.path.join(target_dir, 'dice_aligner.bat'))

    # Update with encrypted files.
    copy_tree(os.path.join(aligner_encrypted_path, 'aligner_engine'),
              os.path.join(target_dir, 'aligner_engine'))
    copy_tree(os.path.join(aligner_encrypted_path, 'aligner_gui'),
              os.path.join(target_dir, 'aligner_gui'))
    copy_tree(os.path.join(aligner_encrypted_path, 'pyarmor_runtime_006148'),
              os.path.join(target_dir, 'pyarmor_runtime_006148'))
    shutil.rmtree(aligner_encrypted_path)

    # Update with plain files.
    copy_tree(os.path.join(util.ROOT_PATH, 'aligner_engine', 'mm_rotate_det', 'dice', 'configs'),
              os.path.join(target_dir, 'aligner_engine', 'mm_rotate_det', 'dice', 'configs'))

    delete_all_pycache(os.path.join(target_dir, 'aligner_engine'))
    delete_all_pycache(os.path.join(target_dir, 'aligner_gui'))
    print("All files are ready.")

    print("Copying .venv ...")
    copy_venv(os.path.join(util.ROOT_PATH,'.venv'), os.path.join(target_dir, '.venv'))


if __name__ == "__main__":
    target_dir = sys.argv[1]
    target_dir = os.path.join(target_dir, 'DICE_ALIGNER')
    release(target_dir)
    print("Release succeeded in " + target_dir)
