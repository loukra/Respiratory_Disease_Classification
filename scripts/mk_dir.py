import os

def mk_dir(bi: bool=True, folder:str="Respiratory_Disease_Classification/"):
    """make new data/image directory for saving chunk images

    Args:
        bi (bool, optional): True -> binary classification. Defaults to True.
        folder (str, optional): "workspace directory name" Defaults to "Respiratory_Disease_Classification/".
    """

    # get workspace dir
    try: 
        # get workspace dir
        str = os.getcwd()
        parent_dir = str.split(folder, 1)[0]+folder+"data/image"
        
        os.mkdir(parent_dir)

        if bi: 
            dir_cls = "cls_2"
            dir_test = dir_cls+"/test"
            dir_train = dir_cls+"/train"
            path_cls = os.path.join(parent_dir,dir_cls)
            path_test = os.path.join(parent_dir,dir_test)
            path_train = os.path.join(parent_dir,dir_train)
            os.mkdir(path_cls)
            os.mkdir(path_train)
            os.mkdir(path_test)


        else:
            dir_cls = "cls_3"
            dir_test = dir_cls+"/test"
            dir_train = dir_cls+"/train"
            path_cls = os.path.join(parent_dir,dir_cls)
            path_test = os.path.join(parent_dir,dir_test)
            path_train = os.path.join(parent_dir,dir_train)
            os.mkdir(path_train)
            os.mkdir(path_test)
                
    except OSError as error: #  FileExistsError
        print("The directory already exists")


