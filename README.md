# CALText
This repository contains the source code for CALText model introduced in "CALText: Contextual Attention Localization for Offline Handwritten Text" paper.
The details of this model are presented in:  (Add paper link)

![image](https://user-images.githubusercontent.com/46027794/139389185-14b0c864-b740-4063-b350-b30798a6a4ba.png) ![image](https://user-images.githubusercontent.com/46027794/139389407-7e8fb63e-6259-49fa-8cbc-7cfb2de6b969.png)







Samples of the datasets that were used to train and test the model can be found at: http://faculty.pucit.edu.pk/nazarkhan/work/urdu_ohtr/pucit_ohul_dataset.html


The code in this model was based on the work of:

https://github.com/JianshuZhang/WAP.

https://github.com/wwjwhen/Watch-Attend-and-Parse-tensorflow-version.

# Requirements

Python 3
Tensorflow v1.6


# Usage 

Upload data files into your Colab account, create pickle files (train, valid, and test images and labels) from the dataset. You can place the pickle dataset files at any folder of your preference but change the path settings in the code where this data is bbeing loaded.

Run "makepickle.ipynb" to create pickle files for train and test data. Further distribute the train pickle file into train and valid pickle files by using last 907 images and labels of train as valid.

For training, set mode="train", and run "CALText.ipynb".

For testing, set mode="test", and run "CALText.ipynb".

For Contextual Attention, set alpha_reg=0, while training and testing.

For Contextual Attention Localization, set alpha_reg=1, while training and testing.


# Run on Python Compiler

To run the code on python compiler, copy the code and make file as "makepickle.py" and "CALText.py". Use following commands to run code files.

python makepickle.py

python CALText.py



# Run on Google Colab

Open "makepickle.ipynb" and "CALText.ipynb" notebook in Google Colab Notebook, and run.

Run "%tensorflow_version 1.x" command at colab notebook before running of "CALText.ipynb". 

Change runtime to GPU or TPU for better performance.


Add these lines in notebook for accessing data from google derive: 

from google.colab import drive

drive.mount("/gdrive", force_remount=True)




# References

http://faculty.pucit.edu.pk/nazarkhan/work/urdu_ohtr/pucit_ohul_dataset.html

http://faculty.pucit.edu.pk/nazarkhan/work/urdu_ohtr/index.html

http://faculty.pucit.edu.pk/nazarkhan/work/urdu_ohtr/ICFHR2020_manuscript.pdf




