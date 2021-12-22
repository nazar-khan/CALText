# CALText
This repository contains the source code for CALText model introduced in

"CALText: Contextual Attention Localization for Offline Handwritten Text"
https://arxiv.org/abs/2111.03952

![image](https://user-images.githubusercontent.com/46027794/139389185-14b0c864-b740-4063-b350-b30798a6a4ba.png) ![image](https://user-images.githubusercontent.com/46027794/139389407-7e8fb63e-6259-49fa-8cbc-7cfb2de6b969.png)







Dataset used to train and test the model can be found at http://faculty.pucit.edu.pk/nazarkhan/work/urdu_ohtr/pucit_ohul_dataset.html


The code in this model was based on the work of:

https://github.com/JianshuZhang/WAP.

https://github.com/wwjwhen/Watch-Attend-and-Parse-tensorflow-version.

# Requirements

Python 3
Tensorflow v1.6


# How to train on your own dataset

Place all training line images in the folder "train_lines"

Place all testing line images in the folder "test_lines"

Place all training ground-truth labels in the file "train_labels.xlsx"

Place all testing ground-truth labels in the file "test_labels.xlsx"

Run "makepickle.ipynb" after specifying validation indices (if required). This will place all training, testing and validation images and labels in pickle format in the 'data/' folder

For training, set mode="train", and run "CALText.ipynb". This will place the trained model(s) in 'models/' folder.

For testing, set mode="test", set path of the model to be used and run "CALText.ipynb".

For Contextual Attention, set alpha_reg=0, while training and testing.

For Contextual Attention Localization, set alpha_reg=1, while training and testing.



# Running on Local Machine

To run the code locally, copy the code from the .ipynb notebooks into "makepickle.py" and "CALText.py". Use following commands to run the code files.

python makepickle.py

python CALText.py



# Running on Google Colab

Open "makepickle.ipynb" and "CALText.ipynb" notebook in Google Colab Notebook, and run.

For newer versions, make sure to run "%tensorflow_version 1.x" command before running of "CALText.ipynb".

Change runtime to GPU or TPU for better performance.


Add these lines in notebook for accessing data from Google Drive: 

`from google.colab import drive`

`drive.mount("/gdrive", force_remount=True)`


# References

PUCIT Offline Handwritten Urdu Lines (PUCIT-OHUL) Dataset: http://faculty.pucit.edu.pk/nazarkhan/work/urdu_ohtr/pucit_ohul_dataset.html

<pre>
@article{anjum_caltext_2021,
  author    = {Tayaba Anjum and Nazar Khan},
  title     = {{CALText}: Contextual Attention Localization for Offline Handwritten Text},
  journal   = {CoRR},
  volume    = {abs/2111.03952},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.03952},
  eprinttype = {arXiv},
  eprint    = {2111.03952},
}
</pre>

<pre>
@inproceedings{anjum2020urdu_ohtr,
    author = {Anjum, Tayaba and Khan, Nazar},
    title = {{An attention based method for offline handwritten Urdu text recognition}},
    booktitle = {International Conference on Frontiers in Handwriting Recognition (ICFHR)},
    month = {September},
    year = {2020},
}
</pre>

Previous Work:

http://faculty.pucit.edu.pk/nazarkhan/work/urdu_ohtr/index.html
http://faculty.pucit.edu.pk/nazarkhan/work/urdu_ohtr/ICFHR2020_manuscript.pdf




