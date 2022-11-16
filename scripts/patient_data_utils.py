import os
import re
from collections import Counter
import random 
import shutil 

def patient_train_val_split(scr_folder: str, val_proportion: int = 0.25, seed: int = 30, underpresented_class = 'health_1', balancing = True):
    """ Creates a train validation split by patient to 

    Args:
        scr_folder (str): The folder were the training + validation data is in
        val_size (int, optional): Percentage of data which goes to validation set. Defaults to 0.25.
        seed (int, optional): Seed of random shuffle the patient list.
        underpresented_class (str, optional): if you now the underpresented class pass it here
        balancing (bool, optional): If True there is a little balancing for training set to give more of the underpresented data to the validation set. 

    """
    train_val_folder = scr_folder + 'train_val'
    train_folder = scr_folder + 'train'
    val_folder = scr_folder + 'val'

    if not os.path.exists(train_val_folder):
        return print('No train_val folder detected\nPlease create a train_val folder before running')

    filelist = os.listdir(train_val_folder)
    
    class_folders = filelist[1:]

    train_size = 0
    val_size = 0
    for cls in class_folders:
        total_length = 0
        filelist = sorted(os.listdir(train_val_folder + '/' + cls))
        print(f'the train_val folder contains {len(filelist)} files for class {cls}')
        patients = [re.findall('[fi]\d{1,4}',x)[0] for x in filelist]
        patient_counts = Counter(patients)
        length = len(filelist)
        val_size_total = int(val_proportion * length)
        #print(f'total validation length will be {val_size_total}')

        patient_keys = list(patient_counts.keys())
        random.seed(seed)
        random.shuffle(patient_keys)

        patient_incl = []
        for patient in patient_keys:
            total_length += patient_counts[patient]
            patient_incl.append(patient)
            if total_length > val_size_total:
                if cls == underpresented_class and balancing:
                    patient_incl = patient_incl[:-1]    
                    total_length -= patient_counts[patient]
                elif not balancing:
                    patient_incl = patient_incl[:-1]    
                    total_length -= patient_counts[patient]
                break
        

        filenames_cls_val = [x for x in filelist if any(id in x for id in patient_incl)]
        filenames_cls_train = [x for x in filelist if not any(id in x for id in patient_incl)]
        print(f'{round(100 * len(filenames_cls_val) / length,1)} % goes to validation set for class {cls}\n')
        train_size += len(filenames_cls_train)
        val_size += len(filenames_cls_val)

        file_path_sr_train = [train_val_folder + '/' + cls + "/" + x for x in filenames_cls_train]
        file_path_tar_train = [train_folder + '/' + cls + "/" + x for x in filenames_cls_train]

        if os.path.exists(train_folder + '/' + cls):
            shutil.rmtree(train_folder + '/' + cls,ignore_errors=True)
        # make new dir
        os.makedirs(train_folder + '/' + cls)
        
        if os.path.exists(val_folder + '/' + cls):
            shutil.rmtree(val_folder + '/' + cls,ignore_errors=True)
        # make new dir
        os.makedirs(val_folder + '/' + cls)

        for src,tar in zip(file_path_sr_train,file_path_tar_train):
            shutil.copy(src, tar)

        file_path_sr_val = [train_val_folder + '/' + cls + "/" + x for x in filenames_cls_val]
        file_path_tar_val = [val_folder + '/' + cls + "/" + x for x in filenames_cls_val]
        for src,tar in zip(file_path_sr_val,file_path_tar_val):
            shutil.copy(src, tar)


    print(f'{round(100 * len(filenames_cls_val)/val_size,1)} % validation set belongs to class {cls}')    
    print(f'{round(100 * len(filenames_cls_train)/train_size,1)} % training set belongs to class {cls}')


