# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:51:17 2023

@author: Annan
"""

import numpy as np
import pickle
import streamlit

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))
