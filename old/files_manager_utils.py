#Utils file incuding function to manage file

import numpy as np
from math import *
from scipy.integrate import quad
import sys
import shutil
import os
import pandas as pd 




def copy_and_rename(src_path, dest_path, new_name):
    #copy a file with path src_path (path/name_of_file) into a new directory dest_path 
    #the file is also renamed using new_name. 
    #!PAY ATTENTION! :  extensions must be included in old and new file names
    shutil.copy(src_path, dest_path)
    new_path = f"{dest_path}/{new_name}"
    shutil.move(f"{dest_path}/{src_path}", new_path)

#-------------------------------------------------------
def find_max_min(csv_file, head = None):
    #takes a .csv file and its header (None by std) 
    #returns the max and the min value of the intire file
    #works also with file with lines of different length
    max_value = float('-inf')
    min_value = float('inf')
    with open (csv_file , 'r') as file:
        for line in file:
            values = line.strip().split(',')    #divide  in single values
            for value in values:
                try: 
                    num = float(value)
                    if num > max_value : max_value = num
                    if num < min_value : min_value = num
                except ValueError: continue   #prevents error (ex Nan or string type if present)

    return max_value, min_value
#-------------------------------------------------------
def csv_line_to_array(csv_file, line_number, head=None):
    #takes a csv file, the number of selected line and the header of the file
    #returns an array filled with the values found in the line
    with open (csv_file, 'r') as file:
        for current_line, line in enumerate(file):
            if current_line == line_number:
                values = line.strip().split(',')
                res_array = []
                for value in values:
                    try:
                        res_array.append(float(value))
         
                    except ValueError : continue
    
                return np.array(res_array)
   
    raise IndexError("Line number out of range")

#-------------------------------------------------------
def get_folder_name(path):
    # Remove trailing slashes
    path = path.rstrip(os.path.sep)
    
    # Return the basename
    return os.path.basename(path)
