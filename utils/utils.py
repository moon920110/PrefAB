import os


def create_new_filename(root, base_name):
    counter = 1
    new_name = base_name

    while os.path.exists(os.path.join(root, new_name)):
        new_name = f'{base_name}_{counter}'
        counter += 1

    return new_name