import os
import re


def has_log_file(log_root):
    file_names = os.listdir(log_root)
    for file_name in file_names:
        if file_name.startswith('log'):
            return True
    return False


def find_max_log(log_root):
    files = os.listdir(log_root)
    pattern = r'log(\d+)\.pth'
    max_num = 0
    for file in files:
        match = re.match(pattern, file)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return os.path.join(log_root, f"log{max_num}.pth")


def classes_txt(root, out_path, num_class=None):
    dirs = os.listdir(root)
    if not num_class:
        num_class = len(dirs)

    with open(out_path, 'w') as f:
        end = 0
        if end < num_class - 1:
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir1 in dirs:
                files = os.listdir(os.path.join(root, dir1))
                for file in files:
                    f.write(os.path.join(root, dir1, file) + '\n')
