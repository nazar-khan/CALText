### To generate pickle files of KHATT dataset###

import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
import cv2
import sys
import argparse

##Only include while running on colab
running_on_colab = True
if running_on_colab == True:
  from google.colab import drive
  drive.mount("/gdrive", force_remount=True)
  dataset_folder = '/gdrive/My Drive/CALTex/dataset/KHATT/'
  data_folder= '/gdrive/My Drive/CALTex/data/KHATT/'
else:
  dataset_folder = 'dataset/KHATT/'
  data_folder= 'data/KHATT/'  

#This function makes a dictionary/vocabulary of all the unique characters in the labels file along with uniquely assigned numeric values to each different character.      
def create_vocabulary(labelfile):
    df = pd.read_excel(labelfile)
    lexicon = {}
    key = 1
    for i in df.index:                               #Iteration over all labels/captions of training images.
      caption = df['Revised'][i]                     #Label/caption of i-th image.
      slen = len(caption)
      j = 0
      while(j<slen):
        ss = caption[j]                                    #Iteration over characters in the label/caption of i-th image. 
        if ss not in lexicon:
            lexicon[ss] = int(key)                   #Set of unique characters with corresponding assigned labels. 
            #print(str(lexicon[ss])+'\t'+ss)
            key = key + 1
        j = j + 1
      i = i + 1
    return lexicon

def save_vocabulary(worddicts):
  ##Save in txt format
  fp=open(dataset_folder + 'vocabulary.txt', 'w')
  worddicts_r = [None] * (len(worddicts)+1)
  i=1
  for kk, vv in worddicts.items():
    if(i<len(worddicts)+1):
    		worddicts_r[vv] = kk
    		fp.write(kk)
    		fp.write('\t')
    		fp.write(str(vv))

    		fp.write('\n')

    else:
    		break
    i=i+1
  fp.close()
  
  ##Save in pickle format
  outFilePtr = open(data_folder + 'vocabulary.pkl','wb')
  pkl.dump(worddicts,outFilePtr)
  outFilePtr.close()



def partition(images, labels, valid_ind):
  train_labels=[]
  train_images={}
  valid_labels=[]
  valid_images={}
  data_part=len(images)-valid_ind
  for i in range(len(images)):
    if i<data_part:
      train_images[i]=images[i]
      train_labels.append(labels[i])
    else:
      valid_images[i-data_part]=images[i]
      valid_labels.append(labels[i])
  return train_images, train_labels, valid_images, valid_labels


#This function loads all the images from the imgfolder and corresponding labels of each image from labelfile.
#According to the dictionary, labels are converted into numeric sequence. 
def load_data(imgfolder,labelfile,dictionary):  
    ImagesLabels = []
    InputImages = {}
    count = 0
    df = pd.read_excel(labelfile)
    split_ch = '-'
    n = 0
    for i in df.index:
      line = df['Num'][i]                             #To read name of i-th image from labelfile. 
      caption = df['Caption'][i]                      #To read label caption of i-th image from labelfile.
      slen=len(caption)
      image_file = imgfolder + line.strip() + '.png'  #To make complete path by appending folder name, image name and image ext. 
      img = cv2.imread(image_file,-1)
      if img is None:
        print(image_file+' not available')
      else:
        if len(img.shape)>2:
          img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)             #Convert to grayscale image.
        
        img=cv2.resize(img, dsize=(800,100), interpolation = cv2.INTER_AREA)        #Image resize.
        
        caption = caption[::-1]	#To handle right-to-left nature of Urdu

        print("----------------------------------------------")
        print(image_file)
        print("Ground-truth string: "+caption)                                 
        
        InputImages[count] = img
        count = count+1
        w_list = []
        w = 0
        while(w < slen):
          ss = caption[w]
          if ss in dictionary:          
            w_list.append(dictionary[ss]) #To access numeric value corresponding to Urdu character from the dictionary.
          w = w + 1        
        xx = w_list[::-1] 	#Take inverse of sequence to align sequence of numeric values with image pixels, as Urdu is read from right to left.
        xx.append(0) 		#0 is appended after each line to represent end of line character. 
        ImagesLabels.append(xx)
        n = n + 1
        print("Ground-truth in numeric representation: "+str(xx).strip('[]'))

        # Set following condition to False in order to load all data without visualizing each image and ground-truth
        if True:
          print("Close the image to see the next image or press Ctrl-C to exit from terminal.")
          plt.imshow(img, cmap="gray")
          plt.title((str(xx).strip('[]')+"\n"+caption), color='b')
          plt.axis('off')
          plt.show()
        
        print("----------------------------------------------")
    print(n)
    return InputImages, ImagesLabels  

def main():
  train_images_path=dataset_folder + 'Train/'
  train_labels_path=dataset_folder + 'TrainingLabels-NoContext.txt'

  valid_images_path=dataset_folder +  'Validate/'
  valid_labels_path=dataset_folder + 'ValidateLabels-NoContext.txt'

  test_images_path=dataset_folder +  'Test/'
  test_labels_path=dataset_folder + 'TestLabels-NoContext.txt'

  #Load dictionary and data.
  #CAUTION: Dictionary/Vocabulary is always made from train_labels.xlsx.
  #Do not change this even when generating a pickle file for testing data.
  worddicts= create_vocabulary(train_labels_path)
  save_vocabulary(worddicts)

  train_images,train_labels = load_data(train_images_path,train_labels_path,worddicts)
  valid_images,valid_labels = load_data(valid_images_path,valid_labels_path,worddicts)
  test_images,test_labels = load_data(test_images_path,test_labels_path,worddicts)

  outFilePtr1 = open(data_folder + 'train_images.pkl','wb')
  outFilePtr2 = open(data_folder + 'train_labels.pkl','wb')
  outFilePtr3 = open(data_folder + 'valid_images.pkl','wb')
  outFilePtr4 = open(data_folder + 'valid_labels.pkl','wb')
  outFilePtr5 = open(data_folder + 'test_images.pkl','wb')
  outFilePtr6 = open(data_folder + 'test_labels.pkl','wb')
  
  pkl.dump(train_images,outFilePtr1)
  pkl.dump(train_labels,outFilePtr2)
  pkl.dump(valid_images,outFilePtr3)
  pkl.dump(valid_labels,outFilePtr4)
  pkl.dump(test_images,outFilePtr5)
  pkl.dump(test_labels,outFilePtr6)

  outFilePtr1.close()
  outFilePtr2.close()
  outFilePtr3.close()
  outFilePtr4.close()
  outFilePtr5.close()
  outFilePtr6.close()


if __name__ == "__main__":
	main()    
