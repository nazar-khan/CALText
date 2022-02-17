running_on_colab = True
if running_on_colab == True:
  ##to upload files at google drive while running using colab
  from google.colab import files
  uploaded = files.upload()
  uploaded = files.upload()
  uploaded = files.upload()
  uploaded = files.upload()

  # only include while running on Google Colab
  from google.colab import drive
  drive.mount('/gdrive', force_remount = True)
  results_folder = '/gdrive/My Drive/CALText/results/'
  model_folder = '/gdrive/My Drive/CALText/model/'
  checkpoints_folder = '/gdrive/My Drive/CALText/checkpoints/'
  data_folder = '/gdrive/My Drive/CALText/data/'

else:
  results_folder = 'results/'
  model_folder = 'model/'
  checkpoints_folder = 'checkpoints/'
  data_folder = 'data/'



import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import pickle as pkl
import cv2
import re
import random
import time
import CALTextModel
import data
import utility
import compute_error
import sys
import argparse


MAXIMAGESIZE=500000
BATCH_IMAGESIZE=500000
BATCHSIZE=4
MAX_LEN=130
BEAM_SIZE=10
NUM_EPOCHS=50

log = open(results_folder+'log.txt', 'w')


def recognize(data,data_uid_list, model,sess, space_ind, dispFreq, datatype):
  fpp_sample = open(results_folder+datatype+'_predicted.txt', 'w')
  fpp_sample2=open(results_folder+datatype+'_target.txt', 'w')
  data_count_idx = 0  
  for batch_x, batch_y in data:
    for idxx,[xx,yy] in enumerate(zip(batch_x,batch_y)): 
    	if(len(xx.shape))>2:
    		xx= cv2.cvtColor(xx.astype('float32'), cv2.COLOR_BGR2GRAY)
    	xx_pad = np.zeros((xx.shape[0], xx.shape[1]), dtype='float32')
    	xx_pad[:xx.shape[0],:xx.shape[1]] = (xx-xx.min())/(xx.max()-xx.min())
    	xx_pad = xx_pad[None, :, :]

    	sequence,_=CALTextModel.model_infer(model, sess, xx_pad, MAX_LEN, BEAM_SIZE)  
    	fpp_sample.write(str(data_uid_list[data_count_idx])+'im')
    	fpp_sample2.write(str(data_uid_list[data_count_idx])+'im')
    	data_count_idx=data_count_idx+1
    	if np.mod(data_count_idx, dispFreq) == 0:
    		print('gen %d samples'%data_count_idx)
    	log.write('gen %d samples'%data_count_idx + '\n')
    	log.flush()
    	for vv in sequence:
    		if vv == 0: # <eol>
    			break
    		fpp_sample.write(' '+str(vv))
    	for y1 in yy:
    		fpp_sample2.write(' '+str(y1))
    	fpp_sample.write('\n')
    	fpp_sample2.write('\n')
  fpp_sample.close()
  fpp_sample2.close()
  compute_error.process( results_folder+datatype+'_predicted.txt', results_folder+datatype+'_target.txt' , results_folder+datatype+'-wer.wer',space_ind)
  fpp=open(results_folder+datatype+'-wer.wer')
  stuff=fpp.readlines()
  fpp.close()
  m=re.search('CER (.*)\n',stuff[0])
  data_cer=100. * float(m.group(1))
  m=re.search('WER (.*)\n',stuff[1])
  data_wer=100. * float(m.group(1))
  return data_cer, data_wer


def predict_probability(data, model,sess):
  probs = []
  for batch_x, batch_y in data:
    	batch_x, batch_x_m, batch_y, batch_y_m = data.prepare_data(batch_x, batch_y,BATCHSIZE)
    	pprobes=CALTextModel.model_getcost(model, sess, batch_x, batch_x_m, batch_y, batch_y_m) 
    	probs.append(pprobs)
  data_errs = np.array(probs)
  data_err_cost = data_errs.mean()



def main(args):

  if args.dataset == 'PUCIT_OHUL':
    data_folder_path=data_folder + 'PUCIT_OHUL/'
  if args.dataset == 'KHATT':
    data_folder_path=data_folder + 'KHATT/' 
  #### Loding Data ####
  train,train_uid_list = data.dataIterator(data_folder_path+'train_lines.pkl',data_folder_path+'train_labels.pkl', batch_size=BATCHSIZE, batch_Imagesize=BATCH_IMAGESIZE,maxlen=MAX_LEN, maxImagesize=MAXIMAGESIZE)
  valid, valid_uid_list = data.dataIterator(data_folder_path+'valid_lines.pkl',data_folder_path+'valid_labels.pkl',batch_size=BATCHSIZE, batch_Imagesize=BATCH_IMAGESIZE,maxlen=MAX_LEN, maxImagesize=MAXIMAGESIZE)
  test, test_uid_list = data.dataIterator(data_folder_path+'test_lines.pkl',data_folder_path+'test_labels.pkl',batch_size=1, batch_Imagesize=BATCH_IMAGESIZE,maxlen=MAX_LEN, maxImagesize=MAXIMAGESIZE)
  worddicts, vocabulary_count, space_ind=utility.load_dict_picklefile(data_folder_path+'vocabulary.pkl')
  num_classes=vocabulary_count+1

  with tf.Graph().as_default():
    	model = CALTextModel.Model()
    	model.build_model(num_classes)
    	config = tf.ConfigProto()
    	config.gpu_options.allow_growth=True
    	init = tf.global_variables_initializer()
    	saver = tf.train.Saver()
    	checkpoint_path = checkpoints_folder+ 'cp-{ckpt:04d}-{acc:04f}.ckpt'
    	model_path = model_folder+ 'current_best_model.ckpt'

    	with tf.Session(config=config) as sess:
    		uidx = 0
    		cost_s = 0
    		cost_i=0
    		dispFreq = 200
    		history_errs = []
    		estop = False
    		halfLrFlag = 0
    		patience = 15
    		lrate = 1  
    		sess.run(init)
    		if (args.mode == 'train'):
    			pre_valid_cer=100
    			for epoch in range(1,NUM_EPOCHS):
    				random.shuffle(train)
    				start_time = time.time()
    				valid_err=0
    				for batch_x, batch_y in train:
    					batch_x, batch_x_m, batch_y, batch_y_m = data.prepare_data(batch_x, batch_y,BATCHSIZE)         
    					uidx += 1
    					#cost_i=CALTextModel.model_train(model, sess, batch_x,batch_y,batch_x_m,batch_y_m,lrate, args.alpha_reg) 
    					cost_s +=0#cost_i

    					if np.isnan(cost_i) or np.isinf(cost_i):
    						print('invalid cost value detected')
    						sys.exit(0)
					 
    					if np.mod(uidx, dispFreq) == 0:
    						cost_s /= dispFreq
    						print('Epoch ', epoch, 'Update ', uidx, 'Cost ', cost_s, 'Lr ', lrate)
    						log.write('Epoch ' + str(epoch) + ' Update ' + str(uidx) + ' Cost ' + str(cost_s) + ' Lr ' + str(lrate) + '\n')
    						log.flush()
    						cost_s = 0

    					if np.mod(uidx,len(train)) == 0:
    						valid_cer, valid_wer=recognize(valid,valid_uid_list, model,sess,space_ind, dispFreq,'valid')
    						valid_err=valid_cer
    						print('valid set decode done')
    						log.write('valid set decode done\n')
    						log.flush()
					 
    					if np.mod(uidx, len(train)) == 0:
    						valid_err_cost=predict_probability(valid, model, sess)
    						history_errs.append(valid_err)
    						if uidx/validFreq == 0 or valid_err <= np.array(history_errs).min():
    							bad_counter = 0
    						if uidx/validFreq != 0 and valid_err > np.array(history_errs).min():
    							bad_counter += 1
    						if bad_counter > patience:
    							if halfLrFlag==2:
    								print('Early Stop!')
    								log.write('Early Stop!\n')
    								log.flush()
    								estop = True
    								break
    							else:
    								print('Lr decay and retrain!')
    								log.write('Lr decay and retrain!\n')
    								log.flush()
    								bad_counter = 0
    								lrate = lrate / 10
    								halfLrFlag += 1
	
    						print('Valid CER: %.2f%%,Valid WER: %.2f%%, Cost: %f' % (valid_cer,valid_wer,valid_err_cost))
    						log.write('Valid CER: %.2f%%,Valid WER: %.2f%%, Cost: %f' % (valid_cer,valid_wer,valid_err_cost) + '\n')
    						log.flush()

    						duration = time.time() - start_time
    						print('1 itr completion time: %.2f%%' %(duration)+'\n')
    						if(valid_cer < pre_valid_cer):
    							pre_valid_cer=valid_cer
    							save_path = saver.save(sess, model_path.format(acc=valid_cer))
    						save_path = saver.save(sess, checkpoint_path.format(ckpt=epoch,acc=valid_cer))
    				if estop:
    					break
						 
    	   ###Testing mode ###			
    		if(args.mode=='test'):
    			saver.restore(sess, model_folder + 'current_best_model.ckpt') 
    			test_cer, test_wer=recognize(test,test_uid_list, model,sess, space_ind, dispFreq ,'test')    		  			
    			print('test CER: %.2f%%,test WER: %.2f%%' % (test_cer,test_wer))
    			log.write('test CER: %.2f%%,test WER: %.2f%%' % (test_cer,test_wer) + '\n')
    			log.flush()		   


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default='train')
	parser.add_argument("--dataset", default='PUCIT_OHUL')
	parser.add_argument("--alpha_reg", type=int, default=1)
	(args, unknown) = parser.parse_known_args()
	main(args)
