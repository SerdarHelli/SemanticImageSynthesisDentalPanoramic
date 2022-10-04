# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:46:05 2022

@author: Burhan
"""
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import os
from tkinter import ttk
import pandas as pd
import numpy as np
from time import time
data = {}
absolute_path = os.path.dirname(__file__)
relative_path = "Gaugan Results wtih abnormality"
save_rel = "Saved_Results.csv"

save_path = os.path.join(absolute_path, save_rel)
dir_path = os.path.join(absolute_path, relative_path)

dirs=  [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and  f.endswith(".png") or f.endswith(".jpg") ]


# if os.path.isfile(save_path):
#     results = pd.read_csv(save_path)
#     if 'Unnamed: 0' in results.columns:
#         results.drop('Unnamed: 0', axis=1, inplace=True)
#     else:
#         results =pd.DataFrame(columns=[ 'Image','FakeOrNot','Elapsed Time(sec)','Elapsed Time(min)'])
results =pd.DataFrame(columns=[ 'Image','FakeOrNot','Elapsed Time(sec)','Elapsed Time(min)'])
# dirs_it=iter(dirs)
class ExampleView(tk.Frame):
    def __init__(self, root,save_path,dirs,results,start):
        self.save_path = save_path
        self.results=results
        self.dirs=dirs
        self.start = start
        self.timer = 0
        self.count=0

        tk.Frame.__init__(self, root)
        
        self.img = ImageTk.PhotoImage(Image.open("C:/Users/Burhan/Desktop/Gaugan Results/Gaugan Results wtih abnormality/fake_image_000.png"))
        panel = Label(root, image = self.img)               #tkinter is bugged thus images are get lost in local scope!
        # panel.image = img                                 #Same with the aformentioned, different solution.
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        generated_button = ttk.Button(text = "Genareted",command=self.increment_counter_generated())
        generated_button.pack(side="bottom", fill="both", expand=True)
        real_button=ttk.Button(text="Real",command=self.increment_counter_real())
        real_button.pack(side="bottom", fill="both", expand=True)

    def save(self):
        self.results.to_csv(self.save_path)  #?

    def endlog(self,start):
          end = time()
          self.timer = end-self.start
    
    def getTime(self):
        return(self.timer)
    
    def get_counter(self):
        return(np.int(self.count))
    
    def increment_counter_generated(self):
        self.endlog(start)
        data={'Image': self.dirs[self.get_counter()], 'FakeOrNot': 0,'Elapsed Time(sec)':self.getTime(),'Elapsed Time(min)':(self.getTime()/60)}
        self.results=self.results.append(data,ignore_index=True)
        self.count += 1
    
    def increment_counter_real(self):
        self.endlog(start)
        data={'Image': self.dirs[self.get_counter()], 'FakeOrNot': 1,'Elapsed Time(sec)':self.getTime(),'Elapsed Time(min)':(self.getTime()/60)}
        self.results=self.results.append(data,ignore_index=True)
        self.count += 1
        
    def get_resultslist(self):
        return self.results["Image"].values
    
    def get_invalid_counter(self):
        for i in range(self.count,len(self.dirs)):
            if not self.dirs[i] in self.get_resultslist():
                break
        return i
   
        
start=0        
root = tk.Tk()        
ExampleView = ExampleView(root,save_path,results,dirs,start)      
if len(dirs)>(ExampleView.get_counter()):
    if dirs[ExampleView.get_counter()] in ExampleView.get_resultslist():
        start = time() 
        view = ExampleView(root,save_path,results,dirs,start)
        view.pack(side="top", fill="both", expand=True)
        root.mainloop()
    