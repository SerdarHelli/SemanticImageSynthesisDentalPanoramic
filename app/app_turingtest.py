# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 16:45:19 2022

@author: Burhan
"""
# path = 'C:/Users/Burhan/Desktop/Gaugan Results/Gaugan Results wtih abnormality/'
#C:/Users/Burhan/Desktop/Gaugan Results/Streamlit_App.py
import streamlit as st
import atexit
import os
from PIL import Image
import pandas as pd

import numpy as np
from time import time, strftime, localtime
from datetime import timedelta


absolute_path = os.path.dirname(__file__)
relative_path = "Gaugan Results wtih abnormality"
save_rel = "Saved_Results.csv"

save_path = os.path.join(absolute_path, save_rel)
dirname = os.path.join(absolute_path, relative_path)


if 'dir_path' not in st.session_state:
    st.session_state.dir_path = ''
    
st.session_state.dir_path=dirname

    
if 'count' not in st.session_state:
        st.session_state.count = 0
        
if 'results' not in st.session_state:
     if os.path.isfile(save_path):
         st.session_state.results = pd.read_csv(save_path)
         if 'Unnamed: 0' in st.session_state.results.columns:
                st.session_state.results.drop('Unnamed: 0', axis=1, inplace=True)
     else:
         st.session_state.results =pd.DataFrame(columns=[ 'Image','FakeOrNot','Elapsed Time(sec)','Elapsed Time(min)'])

if 'timer' not in st.session_state:
    st.session_state.timer = 0
    
def save():
    st.session_state.results.to_csv(save_path)



# import glob
# import tkinter as tk
# from tkinter import filedialog
# root = tk.Tk()
# root.withdraw()
# root.wm_attributes('-topmost', 1)



    #Folder Picker
    # if st.session_state.dir_path == '':
    #     st.write('Please select a folder:')
    #     clicked = st.button('Folder Picker')
#     if clicked:
#         dirname = st.text_input('Selected folder:', filedialog.askdirectory(master=root))
#         st.session_state.dir_path=dirname

dirs=  [f for f in os.listdir(st.session_state.dir_path) if os.path.isfile(os.path.join(st.session_state.dir_path, f)) and  f.endswith(".png") or f.endswith(".jpg") ]

if 'done' not in st.session_state:
    st.session_state.done =0

def endlog():
    end = time()
    st.session_state.timer=end-start

def getTime():
    return(st.session_state.timer)
    
def increment_counter_genareted():
    endlog()
    data={'Image': dirs[get_counter()], 'FakeOrNot': 0,'Elapsed Time(sec)':getTime(),'Elapsed Time(min)':(getTime()/60)}
    st.session_state.results=st.session_state.results.append(data,ignore_index=True)
    st.session_state.count += 1
    
def get_counter():
    return np.int(st.session_state.count)

def increment_counter_real():
    endlog()
    data={'Image': dirs[get_counter()], 'FakeOrNot': 1,'Elapsed Time(sec)':getTime(),'Elapsed Time(min)':(getTime()/60)}
    st.session_state.results=st.session_state.results.append(data,ignore_index=True)
    st.session_state.count += 1
    
def get_resultslist():
    return st.session_state.results["Image"].values

def get_invalid_counter():
    for i in range(st.session_state.count,len(dirs)):
        if not dirs[i] in get_resultslist():
            break
    return i
    
        

        

if len(dirs)>(get_counter()):

    if dirs[get_counter()] in get_resultslist():
        st.session_state.count=get_invalid_counter()
    start = time()    
    image_path=os.path.join(st.session_state.dir_path, dirs[get_counter()])
    image=Image.open(image_path)
    st.image(image)
    generated_button=st.button("Genareted",on_click=increment_counter_genareted)    
    real_button=st.button("Real",on_click=increment_counter_real)
    if st.button("Please press to save your progress",on_click=save):
        st.write("Your progress has been saved!")
else:
    st.write("Done!")
    
    
st.write(st.session_state.count)

    
st.dataframe(st.session_state.results)








