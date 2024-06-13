import pandas as pd
import numpy as np
import os
import shutil

def create_dataset():
    populated_dirs = []
    source_dir = 'data/data_raw/'
    dest_dir = 'data/'
    annotation_dir = 'data/annotations'

    class_names = pd.read_csv('data/annotations/class_list.txt', header=None, sep=' ')
    class_names_dict = class_names.set_index(0).T.to_dict('records')[0]
    steps = [("train_info.csv", "train_set", "train"),
            ("val_info.csv", "val_set", "test")]

    for triplet in steps:
        annot_info, input_set, output_folder = triplet

        annot_path = os.path.join(annotation_dir, annot_info)
        input_path = os.path.join(source_dir, input_set)
        output_path =  os.path.join(dest_dir, output_folder)

        df = pd.read_csv(annot_path, header= None)
        df.columns = ['filename', 'class_num']
        df['class_name'] = df['class_num'].map(class_names_dict)

        # Create the destination directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Directory {output_path} created")
        else:
            print(f"Populating {output_path}")
        # Iterate over the dataframe and copy files to the appropriate folder
        for index, row in df.iterrows():
            class_folder = os.path.join(output_path, row['class_name'])
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            source_file = os.path.join(input_path, row['filename'])
            dest_file = os.path.join(class_folder, row['filename'])
            if os.path.exists(source_file):
                shutil.copy(source_file, dest_file)

        directory_structure = {}
        for class_name in df['class_name'].unique():
            class_folder = os.path.join(output_path, class_name)
            directory_structure[class_name] = os.listdir(class_folder)
        populated_dirs.append(directory_structure)

    return populated_dirs

def create_validation(random_state = 42):
    np.random.seed(random_state)
    source_dir = 'data/train'
    dest_dir = 'data/val'
    val_size = 0.2
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for class_name in os.listdir(source_dir):
        class_folder = os.path.join(source_dir, class_name)
        dest_class_folder = os.path.join(dest_dir, class_name)
        if not os.path.exists(dest_class_folder):
            os.makedirs(dest_class_folder)
        files = os.listdir(class_folder)
        val_files = np.random.choice(files, int(len(files) * val_size), replace=False)
        for file in val_files:
            source_file = os.path.join(class_folder, file)
            dest_file = os.path.join(dest_class_folder, file)
            shutil.move(source_file, dest_file)
    return os.listdir(dest_dir)