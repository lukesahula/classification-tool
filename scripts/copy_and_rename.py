import os
import shutil

def copy_and_rename():
    target_path = os.path.join(os.getcwd(), 'test_t', 'pos')
    path = os.path.join(os.getcwd(), '20170118', 'pos')
    for filename in os.listdir(path):
        if filename.endswith('.gz'):
            old_name = os.path.join(path, filename)
            new_name = os.path.join(target_path, '20170118' + filename)
            shutil.copy(old_name, new_name)

    path = os.path.join(os.getcwd(), '20170125', 'pos')
    for filename in os.listdir(path):
        if filename.endswith('.gz'):
            old_name = os.path.join(path, filename)
            new_name = os.path.join(target_path, '20170125' + filename)
            shutil.copy(old_name, new_name)

    path = os.path.join(os.getcwd(), '20170301', 'pos')
    for filename in os.listdir(path):
        if filename.endswith('.gz'):
            old_name = os.path.join(path, filename)
            new_name = os.path.join(target_path, '20170301' + filename)
            shutil.copy(old_name, new_name)

copy_and_rename()
