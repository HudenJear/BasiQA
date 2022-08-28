import os
import  pandas as pd

def csv_write(data_frame, file_path, params=None, auto_mkdir=True):
    """Write csv to file.

    Args:
        data_frame (pd.DataFrame): csv data.
        file_path (str): saving file path.
        params (None or list): Same as to_csv() interference.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    sav = data_frame.to_csv(file_path)
