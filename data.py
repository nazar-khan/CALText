import pickle as pkl
import numpy as np
import cv2

def dataIterator(feature_file,label_file,batch_size,batch_Imagesize,maxlen,maxImagesize):
    
    fp=open(feature_file,'rb')
    features=pkl.load(fp)
    fp.close()

    fp2=open(label_file,'rb')
    labels=pkl.load(fp2)
    fp2.close()
    

    imageSize={}
    for uid,fea in features.items():
        
        imageSize[uid]=fea.shape[0]*fea.shape[1]

    imageSize= sorted(imageSize.items(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element

    feature_batch=[]
    label_batch=[]
    feature_total=[]
    label_total=[]
    uidList=[]

    batch_image_size=0
    biggest_image_size=0
    i=0
    for uid,size in imageSize:
        if size>biggest_image_size:
            biggest_image_size=size
        fea=features[uid]
        lab=labels[uid]
        batch_image_size=biggest_image_size*(i+1)
        if len(lab)>maxlen:
            print('sentence', uid, 'length bigger than', maxlen, 'ignore')
        elif size>maxImagesize:
            print('image', uid, 'size bigger than', maxImagesize, 'ignore')
        else:
            uidList.append(uid)
            if batch_image_size>batch_Imagesize or i==batch_size: # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)

                i=0
                biggest_image_size=size
                feature_batch=[]
                label_batch=[]
                feature_batch.append(fea)
                label_batch.append(lab)
                batch_image_size=biggest_image_size*(i+1)
                i+=1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i+=1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)

    print('total ',len(feature_total), 'batch data loaded')

    return list(zip(feature_total,label_total)),uidList

def prepare_data(images_x, seqs_y, n_words_src=30000,
                 n_words=30000):

    heights_x = [s.shape[0] for s in images_x]
    widths_x = [s.shape[1] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]

    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_y = np.max(lengths_y) + 1
    x = np.zeros((n_samples, max_height_x, max_width_x)).astype('float32')
    
    y =np.zeros((maxlen_y, n_samples)).astype('int64') # the <eol> must be 0 in the dict !!!
    x_mask =np.zeros((n_samples, max_height_x, max_width_x)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
 
    for idx, [s_x, s_y] in enumerate(zip(images_x, seqs_y)):

        #s_x=(s_x / 255.) # [B, C, H, W] -> [B, H, W, C]       
        if len(s_x.shape)>2:
          s_x= cv2.cvtColor(s_x.astype('float32'), cv2.COLOR_BGR2GRAY)
        x[idx, :heights_x[idx], :widths_x[idx]] = (s_x-s_x.min())/(s_x.max()-s_x.min())
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx], idx] = 1.
    return x, x_mask, y, y_mask

def preprocess_img(img):       
    if len(img.shape)>2:
      img= cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
    height=img.shape[0]
    width=img.shape[1]

    if(width<300):
      result = np.ones([img.shape[0], img.shape[1]*2])*255
      result[0:img.shape[0],img.shape[1]:img.shape[1]*2]=img
      img=result
    if(height>300):
      img=img[0:300,:]
    img=cv2.resize(img, dsize=(800,100), interpolation = cv2.INTER_AREA)

    xx_pad = np.zeros((img.shape[0], img.shape[1]), dtype='float32')			
    xx_pad[:,:] =(img-img.min())/(img.max()-img.min())				
    xx_pad = xx_pad[None, :, :]
    return img, xx_pad
