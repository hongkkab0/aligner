import subprocess
import os


def get_file_list(dir_name):
    file_list = os.listdir(dir_name)
    all_files = list()
    for entry in file_list:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_file_list(full_path)
        else:
            all_files.append(full_path)

    return all_files


def get_ui_list(path='.'):
    file_list = get_file_list(path)

    ui_list = []
    for elem in file_list:
        if elem.endswith('.ui'):
            ui_list.append(elem)

    return ui_list


# def run_process(input_name: str):
#     print('run_process - ', input_name)
#     root_path = os.path.dirname(os.path.dirname(__file__))
#
#     output_name = input_name.replace('.ui', '.py')
#     args = [ root_path + '\\.venv\\Scripts\\python', '-m', 'PyQt5.uic.pyuic', '-x', input_name, '-o', output_name]
#     proc = subprocess.run(args)
#     return proc

def run_process(input_name: str):
    print('run_process - ', input_name)
    root_path = os.path.dirname(os.path.dirname(__file__))

    output_name = input_name.replace('.ui', '.py')
    args = [root_path + '\\.venv\\Scripts\\python', '-m', 'PyQt5.uic.pyuic', '-x', input_name, '-o', output_name]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def main():
    ui_list = get_ui_list()
    process_list = []
    for ui in ui_list:
        process = run_process(ui)
        process_list.append(process)

    for process in process_list:
        process.wait()

    print('ui2py - finished!')



if __name__ == '__main__':
    main()
