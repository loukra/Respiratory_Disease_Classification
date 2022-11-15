import random
import shutil
import os
import pandas as pd
import numpy as np

def balance_sel(df:pd.DataFrame, ranseed: int=30): 
    """do a balanced selection between health_0 and health_1 classes in training data. 
       choose all the subject in the minority group, then choose the same subject number from major class.
       cope the selected images to new /balanced folder.

    Args:
        df (pd.DataFrame): annotation file of all the audio files
        ranseed (int, optional): set random seed for shuffle. Defaults to 30.
    """
    # 50-50 clf selection, according to patient
    
    df = df[df.train_test=='train']
    nr_0 = df[df.is_healthy==0].id.nunique()
    nr_1 = df[df.is_healthy==1].id.nunique()
    
    major_cls = np.argmax([nr_0,nr_1])
    minor_cls = np.argmin([nr_0,nr_1])
    major_nr = max(nr_0,nr_1)
    sample_nr = min(nr_0,nr_1)

    #select random samples from majority class
    random.seed(ranseed)
    ran_idx = random.sample(range(major_nr),sample_nr) 

    # random select patient id
    pat_id = df[df.is_healthy==major_cls].id.unique()[ran_idx]

    # look for the images
    source_path = "../../data/images/cls_2/train/health_"+str(major_cls)
    # data/image/balanced/train/health_0
    output_path = "../../data/images/balanced/train/health_"+str(major_cls)
    

    if os.path.exists(output_path):
        shutil.rmtree(output_path,ignore_errors=True)

    os.makedirs(output_path)

    filelist = os.listdir(source_path)
    file_name = [x for x in filelist if any(id in x for id in pat_id)]

    file_path_sr = [source_path + "/" + x for x in file_name]
    file_path_tar = [output_path + "/" + x for x in file_name]

    for src,tar in zip(file_path_sr,file_path_tar):
        shutil.copy(src, tar,)
    print(f"Number of health_{major_cls} class copied: {len(file_path_sr)}.")
   

    # copy minority class folder 
    source_path_mino = "../../data/images/cls_2/train/health_"+str(minor_cls)
    output_path_mino = "../../data/images/balanced/train/health_"+str(minor_cls)
    
    if os.path.exists(output_path_mino):
        shutil.rmtree(output_path_mino,ignore_errors=True)

    os.makedirs(output_path_mino)
    shutil.copytree(source_path_mino, output_path_mino, dirs_exist_ok=True)

