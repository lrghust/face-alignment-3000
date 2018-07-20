#!/bin/bash

#wget https://ibug.doc.ic.ac.uk/download/annotations/300w_cropped.zip
#unzip 300w_cropped.zip
cd 300w_cropped
ls */*.png > full_img_list
cd ..
python train_test_split.py
