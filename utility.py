from skimage.transform import rescale, resize
import numpy as np
import time
import pickle as pkl
import random
rng = np.random.RandomState(int(time.time()))

'''
Following three functions:
norm_weight(),
conv_norm_weight(),
ortho_weight()
are initialization methods for weights.
'''
def norm_weight(fan_in, fan_out):
	W_bound = np.sqrt(6.0 / (fan_in + fan_out))
	return np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(fan_in, fan_out)), dtype=np.float32)
 
def conv_norm_weight(nin, nout, kernel_size):
    filter_shape = (kernel_size[0], kernel_size[1], nin, nout)
    fan_in = kernel_size[0] * kernel_size[1] * nin
    fan_out = kernel_size[0] * kernel_size[1] * nout
    W_bound = np.sqrt(6. / (fan_in + fan_out))
    W = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=np.float32)
    return W.astype('float32')

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


''' This function convert the output sequence into required PUCIT-OHUL/KHATT label's format using dictionary. '''
def decode_sequence(sequence, dictionary):
    ind=1
    outstr=u''
    worddicts_r=dict_ind(dictionary)
    while (ind<len(sequence)-1):
        k=(len(sequence)-1)-ind
        outstr=outstr+worddicts_r[int(sequence[k])]
        ind=ind+1
    return outstr        



'''This function uses color to display effect of time. The attention sequence starts with yellow color and gradully ends with green.''' 
def visualize_temporaly(alpha, seq_len):
    visualization_t=np.zeros((100,800,3), np.float32)
    vis_sum =np.zeros((100,800,3), np.float32)
    color_t=np.zeros((3), np.float32)
    t=0
    while (t<seq_len):
        nt=t/seq_len
        color_t[0] = (1-nt)*float(0) + nt*float(255)
        color_t[1] = (1-nt)*float(255) + nt*float(255)
        color_t[2] = (1-nt)*float(0) + nt*float(0)			
        alpha_t=resize(alpha[t], (100,800))
        visualization_t[:,:,0] = alpha_t * color_t[0]
        visualization_t[:,:,1] = alpha_t * color_t[1]
        visualization_t[:,:,2] = alpha_t * color_t[2]
        vis_sum +=visualization_t
        visualization_t_norm=((visualization_t-visualization_t.min())/(visualization_t.max()-visualization_t.min()))
        t=t+1
    vis_sum_norm=((vis_sum-vis_sum.min())/(vis_sum.max()-vis_sum.min()))
    return vis_sum_norm

def load_dict_picklefile(dictFile):
    fp=open(dictFile,'rb')
    lexicon=pkl.load(fp)
    fp.close()
    return lexicon, len(lexicon), lexicon[' ']

def load_dict_txtfile(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    lex_ind={}
    itr=1
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(itr)
        itr=itr+1
    
    return lexicon,len(lexicon)

def dict_ind(lexicon):
    worddicts_r = [None] * (len(lexicon)+1)
    i=1
    for kk, vv in lexicon.items():
    	if(i<len(lexicon)+1):
    		worddicts_r[vv] = kk
    	else:
    		break
    i=i+1
    return worddicts_r