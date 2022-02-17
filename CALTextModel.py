import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import copy
import math
import utility





'''
DenseEncoder class implements dense encoder 
using bottleneck layers. Encoder contains three dense blocks. 
'''
class DenseEncoder():
	def __init__(self, blocks,       # number of dense blocks
				level,                     # number of levels in each blocks
				growth_rate,               # growth rate in DenseNet paper: k
				training,
				dropout_rate=0.2,          # keep-rate of dropout layer
				dense_channels=0,          # filter numbers of transition layer's input
				transition=0.5,            # rate of comprssion
				input_conv_filters=48,     # filter numbers of conv2d before dense blocks
				input_conv_stride=2,       # stride of conv2d before dense blocks
				input_conv_kernel=[7,7]):  # kernel size of conv2d before dense blocks
		self.blocks = blocks
		self.growth_rate = growth_rate
		self.training = training
		self.dense_channels = dense_channels
		self.level = level
		self.dropout_rate = dropout_rate
		self.transition = transition
		self.input_conv_kernel = input_conv_kernel
		self.input_conv_stride = input_conv_stride
		self.input_conv_filters = input_conv_filters

	def bound(self, nin, nout, kernel):
		fin = nin * kernel[0] * kernel[1]
		fout = nout * kernel[0] * kernel[1]
		return np.sqrt(6. / (fin + fout))
    
	def dense_net(self, input_x, mask_x):

		#### before flowing into dense blocks ####
		input_x=tf.expand_dims(input=input_x, axis=3)
		x = input_x
		limit = self.bound(1, self.input_conv_filters, self.input_conv_kernel)
		x = tf.layers.conv2d(x, filters=self.input_conv_filters, strides=self.input_conv_stride,
			kernel_size=self.input_conv_kernel, padding='SAME', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
		mask_x = mask_x[:, 0::2, 0::2]
		x = tf.layers.batch_normalization(x, training=self.training, momentum=0.9, scale=True, gamma_initializer=tf.random_uniform_initializer(-1.0/math.sqrt(self.input_conv_filters),
				1.0/math.sqrt(self.input_conv_filters), dtype=tf.float32), epsilon=0.0001)
		x = tf.nn.relu(x)
		x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='SAME')
		
		input_pre = x
		mask_x = mask_x[:, 0::2, 0::2]
		self.dense_channels += self.input_conv_filters
		dense_out = x

		#### flowing into dense blocks and transition_layer ####
		for i in range(self.blocks):
			for j in range(self.level):

				#### [1, 1] convolution part for bottleneck ####
				limit = self.bound(self.dense_channels, 4 * self.growth_rate, [1,1])
				x = tf.layers.conv2d(x, filters=4 * self.growth_rate, kernel_size=[1,1],
					strides=1, padding='VALID', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
				x = tf.layers.batch_normalization(inputs=x,  training=self.training, momentum=0.9, scale=True, gamma_initializer=tf.random_uniform_initializer(-1.0/math.sqrt(4 * self.growth_rate),
					1.0/math.sqrt(4 * self.growth_rate), dtype=tf.float32), epsilon=0.0001)
				x = tf.nn.relu(x)
				x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)

				#### [3, 3] convolution part for regular convolve operation
				limit = self.bound(4 * self.growth_rate, self.growth_rate, [3,3])
				x = tf.layers.conv2d(x, filters=self.growth_rate, kernel_size=[3,3],
					strides=1, padding='SAME', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
				x = tf.layers.batch_normalization(inputs=x, training=self.training, momentum=0.9, scale=True, gamma_initializer=tf.random_uniform_initializer(-1.0/math.sqrt(self.growth_rate),
					1.0/math.sqrt(self.growth_rate), dtype=tf.float32), epsilon=0.0001)
				x = tf.nn.relu(x)

				x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)

				dense_out = tf.concat([dense_out, x], axis=3)
				x = dense_out
				#### calculate the filter number of dense block's output ####
				self.dense_channels += self.growth_rate

			if i < self.blocks - 1:
				compressed_channels = int(self.dense_channels * self.transition)

				#### new dense channels for new dense block ####
				self.dense_channels = compressed_channels
				limit = self.bound(self.dense_channels, compressed_channels, [1,1])
				x = tf.layers.conv2d(x, filters=compressed_channels, kernel_size=[1,1],
					strides=1, padding='VALID', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
				x = tf.layers.batch_normalization(x, training=self.training, momentum=0.9, scale=True, gamma_initializer=tf.random_uniform_initializer(-1.0/math.sqrt(self.dense_channels),
						1.0/math.sqrt(self.dense_channels), dtype=tf.float32), epsilon=0.0001)
				x = tf.nn.relu(x)
				x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)
    
				x = tf.layers.average_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='SAME')
				dense_out = x
				mask_x = mask_x[:, 0::2, 0::2]

		return dense_out, mask_x

'''
ContextualAttention class implements contextual attention mechanism. 
'''
class ContextualAttention():
	def __init__(self, channels,                          # output of DenseEncoder | [batch, h, w, channels]
				dim_decoder, dim_attend):                       # decoder hidden state:$h_{t-1}$ | [batch, dec_dim]

		self.channels = channels

		self.coverage_kernel = [11,11]                      # kernel size of $Q$
		self.coverage_filters = dim_attend                  # filter numbers of $Q$ | 512

		self.dim_decoder = dim_decoder                      # 256
		self.dim_attend = dim_attend                        # unified dim of three parts calculating $e_ti$ i.e.
		                                                    # $Q*beta_t$, $U_a * a_i$, $W_a x h_{t-1}$ | 512
		self.U_f = tf.Variable(utility.norm_weight(self.coverage_filters, self.dim_attend), name='U_f') # $U_f x f_i$ | [cov_filters, dim_attend]
		self.U_f_b = tf.Variable(np.zeros((self.dim_attend,)).astype('float32'), name='U_f_b')  # $U_f x f_i + U_f_b$ | [dim_attend, ]

		self.U_a = tf.Variable(utility.norm_weight(self.channels,self.dim_attend), name='U_a')         # $U_a x a_i$ | [annotatin_channels, dim_attend]
		self.U_a_b = tf.Variable(np.zeros((self.dim_attend,)).astype('float32'), name='U_a_b') # $U_a x a_i + U_a_b$ | [dim_attend, ]

		self.W_a = tf.Variable(utility.norm_weight(self.dim_decoder,self.dim_attend), name='W_a')      # $W_a x h_{t_1}$ | [dec_dim, dim_attend]
		self.W_a_b = tf.Variable(np.zeros((self.dim_attend,)).astype('float32'), name='W_a_b') # $W_a x h_{t-1} + W_a_b$ | [dim_attend, ]

		self.V_a = tf.Variable(utility.norm_weight(self.dim_attend, 1), name='V_a')                    # $V_a x tanh(A + B + C)$ | [dim_attend, 1]
		self.V_a_b = tf.Variable(np.zeros((1,)).astype('float32'), name='V_a_b')               # $V_a x tanh(A + B + C) + V_a_b$ | [1, ]

		self.alpha_past_filter = tf.Variable(utility.conv_norm_weight(1, self.dim_attend, self.coverage_kernel), name='alpha_past_filter')


	def get_context(self, annotation4ctx, h_t_1, alpha_past4ctx, a_mask):

		#### calculate $U_f x f_i$ ####
		alpha_past_4d = alpha_past4ctx[:, :, :, None]

		Ft = tf.nn.conv2d(alpha_past_4d, filter=self.alpha_past_filter, strides=[1, 1, 1, 1], padding='SAME')
		coverage_vector = tf.tensordot(Ft, self.U_f, axes=1) 	+ self.U_f_b    # [batch, h, w, dim_attend]

		#### calculate $U_a x a_i$ ####
		dense_encoder_vector = tf.tensordot(annotation4ctx, self.U_a, axes=1) + self.U_a_b   # [batch, h, w, dim_attend]

		#### calculate $W_a x h_{t - 1}$ ####
		speller_vector = tf.tensordot(h_t_1, self.W_a, axes=1) + self.W_a_b   # [batch, dim_attend]
		speller_vector = speller_vector[:, None, None, :]    # [batch, None, None, dim_attend]

		tanh_vector = tf.tanh(coverage_vector + dense_encoder_vector + speller_vector)    # [batch, h, w, dim_attend]
		e_ti = tf.tensordot(tanh_vector, self.V_a, axes=1) + self.V_a_b  # [batch, h, w, 1]
		alpha = tf.exp(e_ti)
		alpha = tf.squeeze(alpha, axis=3)

		if a_mask is not None:
			alpha = alpha * a_mask

		alpha = alpha / tf.reduce_sum(alpha, axis=[1, 2], keepdims=True)    # normlized weights | [batch, h, w]
		alpha_past4ctx += alpha    # accumalated weights matrix | [batch, h, w]
		context = tf.reduce_sum(annotation4ctx * alpha[:, :, :, None], axis=[1, 2])   # context vector | [batch, feature_channels]
		return context, alpha, alpha_past4ctx

'''
Decoder class implements 2 layerd Decoder (GRU) which decodes an input image 
and outputs a seuence of characters using attention mechanism . 
'''
class Decoder():
	def __init__(self, hidden_dim, word_dim, contextual_attention, context_dim):

		self.contextual_attention = contextual_attention                                # inner-instance of contextual_attention to provide context
		self.context_dim = context_dim                          # context dime 684
		self.hidden_dim = hidden_dim                            # dim of hidden state  256
		self.word_dim = word_dim                                # dim of embedding word 256
		
		##GRU 1 weights initialization starts here
		self.W_yz_yr = tf.Variable(np.concatenate(
			[utility.norm_weight(self.word_dim, self.hidden_dim), utility.norm_weight(self.word_dim, self.hidden_dim)], axis=1), name='W_yz_yr') # [dim_word, 2 * dim_decoder]
		self.b_yz_yr = tf.Variable(np.zeros((2 * self.hidden_dim, )).astype('float32'), name='b_yz_yr')

		self.U_hz_hr = tf.Variable(np.concatenate(
			[utility.ortho_weight(self.hidden_dim),utility.ortho_weight(self.hidden_dim)], axis=1), name='U_hz_hr')                              # [dim_hidden, 2 * dim_hidden]

		self.W_yh = tf.Variable(utility.norm_weight(self.word_dim,
			self.hidden_dim), name='W_yh')
		self.b_yh = tf.Variable(np.zeros((self.hidden_dim, )).astype('float32'), name='b_yh')                                    # [dim_decoder, ]

		self.U_rh = tf.Variable(utility.ortho_weight(self.hidden_dim), name='U_rh')                                                      # [dim_hidden, dim_hidden]

		##GRU 2 weights initialization starts here
		self.U_hz_hr_nl = tf.Variable(np.concatenate(
			[utility.ortho_weight(self.hidden_dim), utility.ortho_weight(self.hidden_dim)], axis=1), name='U_hz_hr_nl')                          # [dim_hidden, 2 * dim_hidden] non_linear

		self.b_hz_hr_nl = tf.Variable(np.zeros((2 * self.hidden_dim, )).astype('float32'), name='b_hz_hr_nl')                    # [2 * dim_hidden, ]

		self.W_c_z_r = tf.Variable(utility.norm_weight(self.context_dim,
			2 * self.hidden_dim), name='W_c_z_r')

		self.U_rh_nl = tf.Variable(utility.ortho_weight(self.hidden_dim), name='U_rh_nl')
		self.b_rh_nl = tf.Variable(np.zeros((self.hidden_dim, )).astype('float32'), name='b_rh_nl')

		self.W_c_h_nl = tf.Variable(utility.norm_weight(self.context_dim, self.hidden_dim), name='W_c_h_nl')



	def get_ht_ctx(self, emb_y, target_hidden_state_0, annotations, a_m, y_m):

		res = tf.scan(self.one_time_step, elems=(emb_y, y_m),
			initializer=(target_hidden_state_0,
				tf.zeros([tf.shape(annotations)[0], self.context_dim]),
				tf.zeros([tf.shape(annotations)[0], tf.shape(annotations)[1], tf.shape(annotations)[2]]),
				tf.zeros([tf.shape(annotations)[0], tf.shape(annotations)[1], tf.shape(annotations)[2]]),
				annotations, a_m))

		return res




	def one_time_step(self, tuple_h0_ctx_alpha_alpha_past_annotation, tuple_emb_mask):

		target_hidden_state_0 = tuple_h0_ctx_alpha_alpha_past_annotation[0]
		alpha_past_one        = tuple_h0_ctx_alpha_alpha_past_annotation[3]
		annotation_one        = tuple_h0_ctx_alpha_alpha_past_annotation[4]
		a_mask                = tuple_h0_ctx_alpha_alpha_past_annotation[5]

		emb_y, y_mask = tuple_emb_mask

		#GRU 1 starts here
		emb_y_z_r_vector = tf.tensordot(emb_y, self.W_yz_yr, axes=1) + \
		self.b_yz_yr                                            # [batch, 2 * dim_decoder]
		hidden_z_r_vector = tf.tensordot(target_hidden_state_0,
		self.U_hz_hr, axes=1)                                   # [batch, 2 * dim_decoder]
		pre_z_r_vector = tf.sigmoid(emb_y_z_r_vector + \
		hidden_z_r_vector)                                      # [batch, 2 * dim_decoder]

		r1 = pre_z_r_vector[:, :self.hidden_dim]                # [batch, dim_decoder]
		z1 = pre_z_r_vector[:, self.hidden_dim:]                # [batch, dim_decoder]

		emb_y_h_vector = tf.tensordot(emb_y, self.W_yh, axes=1) + \
		self.b_yh                                               # [batch, dim_decoder]
		hidden_r_h_vector = tf.tensordot(target_hidden_state_0,
		self.U_rh, axes=1)                                      # [batch, dim_decoder]
		hidden_r_h_vector *= r1
		pre_h_proposal = tf.tanh(hidden_r_h_vector + emb_y_h_vector)

		pre_h = z1 * target_hidden_state_0 + (1. - z1) * pre_h_proposal

		if y_mask is not None:
			pre_h = y_mask[:, None] * pre_h + (1. - y_mask)[:, None] * target_hidden_state_0

		context, alpha, alpha_past_one = self.contextual_attention.get_context(annotation_one, pre_h, alpha_past_one, a_mask)  # [batch, dim_ctx]
		
		#GRU 2 starts here
		emb_y_z_r_nl_vector = tf.tensordot(pre_h, self.U_hz_hr_nl, axes=1) + self.b_hz_hr_nl
		context_z_r_vector = tf.tensordot(context, self.W_c_z_r, axes=1)
		z_r_vector = tf.sigmoid(emb_y_z_r_nl_vector + context_z_r_vector)

		r2 = z_r_vector[:, :self.hidden_dim]
		z2 = z_r_vector[:, self.hidden_dim:]

		emb_y_h_nl_vector = tf.tensordot(pre_h, self.U_rh_nl, axes=1) 
		emb_y_h_nl_vector *= r2
		emb_y_h_nl_vector=emb_y_h_nl_vector+ self.b_rh_nl # bias added after point wise multiplication with r2
		context_h_vector = tf.tensordot(context, self.W_c_h_nl, axes=1)
		h_proposal = tf.tanh(emb_y_h_nl_vector + context_h_vector)
		h = z2 * pre_h + (1. - z2) * h_proposal

		if y_mask is not None:
			h = y_mask[:, None] * h + (1. - y_mask)[:, None] * pre_h

		return h, context, alpha, alpha_past_one, annotation_one, a_mask


'''
CALText class is the main class. This class uses below three classes:
1) DenseEncoder (Encoder)
2) ContextualAttention (Contextual attention mechnism)
3) Decoder (2 layerd GRU Decoder)
CALText class implements two functions get_cost and get_sample, which are actually used for cost calculation and decoding.
'''
class CALText():
	def __init__(self, dense_encoder, contextual_attention, decoder, hidden_dim, word_dim, context_dim, target_dim, training):

		#self.batch_size = batch_size
		self.hidden_dim = hidden_dim
		self.word_dim = word_dim
		self.context_dim = context_dim
		self.target_dim = target_dim
		self.embed_matrix = tf.Variable(utility.norm_weight(self.target_dim, self.word_dim), name='embed')

		self.dense_encoder = dense_encoder
		self.contextual_attention = contextual_attention
		self.decoder = decoder
		self.Wa2h = tf.Variable(utility.norm_weight(self.context_dim, self.hidden_dim), name='Wa2h')
		self.ba2h = tf.Variable(np.zeros((self.hidden_dim,)).astype('float32'), name='ba2h')
		self.Wc = tf.Variable(utility.norm_weight(self.context_dim, self.word_dim), name='Wc')
		self.bc = tf.Variable(np.zeros((self.word_dim,)).astype('float32'), name='bc')
		self.Wh = tf.Variable(utility.norm_weight(self.hidden_dim, self.word_dim), name='Wh')
		self.bh = tf.Variable(np.zeros((self.word_dim,)).astype('float32'), name='bh')
		self.Wy = tf.Variable(utility.norm_weight(self.word_dim, self.word_dim), name='Wy')
		self.by = tf.Variable(np.zeros((self.word_dim,)).astype('float32'), name='by')
		self.Wo = tf.Variable(utility.norm_weight(self.word_dim//2, self.target_dim), name='Wo')
		self.bo = tf.Variable(np.zeros((self.target_dim,)).astype('float32'), name='bo')
		self.training = training


	def get_cost(self, cost_annotation, cost_y, a_m, y_m,alpha_reg):
		
		#### step: 1 prepration of embedding of labels sequences #### 
		timesteps = tf.shape(cost_y)[0]
		batch_size = tf.shape(cost_y)[1]
		emb_y = tf.nn.embedding_lookup(self.embed_matrix, tf.reshape(cost_y, [-1]))
		emb_y = tf.reshape(emb_y, [timesteps, batch_size, self.word_dim])
		emb_pad = tf.fill((1, batch_size, self.word_dim), 0.0)
		emb_shift = tf.concat([emb_pad ,tf.strided_slice(emb_y, [0, 0, 0], [-1, batch_size, self.word_dim], [1, 1, 1])], axis=0)
		new_emb_y = emb_shift

		#### step: 2 calculation of h_0 #### 
		anno_mean = tf.reduce_sum(cost_annotation * a_m[:, :, :, None], axis=[1, 2]) / tf.reduce_sum(a_m, axis=[1, 2])[:, None]
		h_0 = tf.tensordot(anno_mean, self.Wa2h, axes=1) + self.ba2h  # [batch, hidden_dim]
		h_0 = tf.tanh(h_0)
	
		#### step: 3 calculation of h_t and c_t at all time steps #### 
		ret = self.decoder.get_ht_ctx(new_emb_y, h_0, cost_annotation, a_m, y_m)
		h_t = ret[0]                      # h_t of all timesteps [timesteps, batch, hidden_dim]
		c_t = ret[1]                      # c_t of all timesteps [timesteps, batch, context_dim]
		alpha=ret[2]											# alpha of all timesteps [timesteps, batch, h, w]

		#### step: 4 calculation of cost using h_t, c_t and y_t_1 ####
		y_t_1 = new_emb_y                 # shifted y | [1:] = [:-1]
		logit_gru = tf.tensordot(h_t, self.Wh, axes=1) + self.bh
		logit_ctx = tf.tensordot(c_t, self.Wc, axes=1) + self.bc
		logit_pre = tf.tensordot(y_t_1, self.Wy, axes=1) + self.by
		logit = logit_pre + logit_ctx + logit_gru
		shape = tf.shape(logit)
		logit = tf.reshape(logit, [shape[0], -1, shape[2]//2, 2])
		logit = tf.reduce_max(logit, axis=3)
		logit = tf.layers.dropout(inputs=logit, rate=0.2, training=self.training)
		logit = tf.tensordot(logit, self.Wo, axes=1) + self.bo
		logit_shape = tf.shape(logit)
		logit = tf.reshape(logit, [-1,logit_shape[2]])
		cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=tf.one_hot(tf.reshape(cost_y, [-1]),depth=self.target_dim))
		
		#### max pooling on vector with size equal to word_dim ####
		cost = tf.multiply(cost, tf.reshape(y_m, [-1]))
		cost = tf.reshape(cost, [shape[0], shape[1]])
		cost = tf.reduce_sum(cost, axis=0)
		cost = tf.reduce_mean(cost)
	
		#### alpha  L1 regularization ####
		alpha_sum=tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(alpha), axis=[2, 3]),axis=0))			
		cost = tf.cond(alpha_reg > 0,  	lambda: cost + (alpha_reg * alpha_sum), lambda: cost)

		return cost


	'''
	The get_word function is called from get_sample function. At first time calll, 
	batch size is equal to 1, later it is called with batch size equal to live_k. 
	'''
	def get_word(self, sample_y, sample_h_pre, alpha_past_pre, sample_annotation):

		emb = tf.cond(sample_y[0] < 0,
			lambda: tf.fill((1, self.word_dim), 0.0),
			lambda: tf.nn.embedding_lookup(self.embed_matrix, sample_y)
			)

		#ret = self.decoder.one_time_step((h_pre, None, None, alpha_past_pre, annotation, None), (emb, None))
		emb_y_z_r_vector = tf.tensordot(emb, self.decoder.W_yz_yr, axes=1) + \
		self.decoder.b_yz_yr                                            # [batch, 2 * dim_decoder]
		hidden_z_r_vector = tf.tensordot(sample_h_pre,
		self.decoder.U_hz_hr, axes=1)                                   # [batch, 2 * dim_decoder]
		pre_z_r_vector = tf.sigmoid(emb_y_z_r_vector + \
		hidden_z_r_vector)                                             # [batch, 2 * dim_decoder]

		r1 = pre_z_r_vector[:, :self.decoder.hidden_dim]                # [batch, dim_decoder]
		z1 = pre_z_r_vector[:, self.decoder.hidden_dim:]                # [batch, dim_decoder]

		emb_y_h_vector = tf.tensordot(emb, self.decoder.W_yh, axes=1) + \
		self.decoder.b_yh                                               # [batch, dim_decoder]
		hidden_r_h_vector = tf.tensordot(sample_h_pre,
		self.decoder.U_rh, axes=1)                                      # [batch, dim_decoder]
		hidden_r_h_vector *= r1
		pre_h_proposal = tf.tanh(hidden_r_h_vector + emb_y_h_vector)

		pre_h = z1 * sample_h_pre + (1. - z1) * pre_h_proposal

		context, contextV, alpha_past = self.decoder.contextual_attention.get_context(sample_annotation, pre_h, alpha_past_pre, None)  # [batch, dim_ctx]
		emb_y_z_r_nl_vector = tf.tensordot(pre_h, self.decoder.U_hz_hr_nl, axes=1) + self.decoder.b_hz_hr_nl
		context_z_r_vector = tf.tensordot(context, self.decoder.W_c_z_r, axes=1)
		z_r_vector = tf.sigmoid(emb_y_z_r_nl_vector + context_z_r_vector)

		r2 = z_r_vector[:, :self.decoder.hidden_dim]
		z2 = z_r_vector[:, self.decoder.hidden_dim:]

		emb_y_h_nl_vector = tf.tensordot(pre_h, self.decoder.U_rh_nl, axes=1) + self.decoder.b_rh_nl
		emb_y_h_nl_vector *= r2
		context_h_vector = tf.tensordot(context, self.decoder.W_c_h_nl, axes=1)
		h_proposal = tf.tanh(emb_y_h_nl_vector + context_h_vector)
		h = z2 * pre_h + (1. - z2) * h_proposal

		h_t = h
		c_t = context
		alpha_past_t = alpha_past
		y_t_1 = emb
		logit_gru = tf.tensordot(h_t, self.Wh, axes=1) + self.bh
		logit_ctx = tf.tensordot(c_t, self.Wc, axes=1) + self.bc
		logit_pre = tf.tensordot(y_t_1, self.Wy, axes=1) + self.by
		logit = logit_pre + logit_ctx + logit_gru   # batch x word_dim

		#### max pooling on vector with size equal to word_dim ####
		shape = tf.shape(logit)
		logit = tf.reshape(logit, [-1, shape[1]//2, 2])
		logit = tf.reduce_max(logit, axis=2)

		logit = tf.layers.dropout(inputs=logit, rate=0.2, training=self.training)

		logit = tf.tensordot(logit, self.Wo, axes=1) + self.bo

		next_probs = tf.nn.softmax(logits=logit)
		next_word  = tf.reduce_max(tf.multinomial(next_probs, num_samples=1), axis=1)
		return next_probs, next_word, h_t, alpha_past_t,contextV 

	'''
	Calculates sequence of labels/characters from annotations using beam search if stochastic is set to false 
	'''

	def get_sample(self,p, w, h, alpha,ctv, ctx0, h_0, k , maxlen, stochastic, session, training):

		sample = []
		sample_score = []
		sample_att=[]

		live_k = 1
		dead_k = 0

		hyp_samples = [[]] * 1
		hyp_scores = np.zeros(live_k).astype('float32')
		hyp_states = []


		next_alpha_past = np.zeros((ctx0.shape[0], ctx0.shape[1], ctx0.shape[2])).astype('float32')
		emb_0 = np.zeros((ctx0.shape[0], 256))

		next_w = -1 * np.ones((1,)).astype('int64')

		next_state = h_0
		for ii in range(maxlen):

			ctx = np.tile(ctx0, [live_k, 1, 1, 1])

			input_dict = {
			anno:ctx,
			infer_y:next_w,
			alpha_past:next_alpha_past,
			h_pre:next_state,
			if_trainning:training
			}
      
			next_p, next_w, next_state, next_alpha_past,contexVec  = session.run([p, w, h, alpha,ctv], feed_dict=input_dict)
			sample_att.append(contexVec[0,:,:])	
	 
			if stochastic:
				
				nw = next_w[0]
				sample.append(nw)
				sample_score += next_p[0, nw]
				if nw == 0:
					break
			else:
				cand_scores = hyp_scores[:, None] - np.log(next_p)
				cand_flat = cand_scores.flatten()
				ranks_flat = cand_flat.argsort()[:(k-dead_k)]
				voc_size = next_p.shape[1]

				assert voc_size==num_classes

				trans_indices = ranks_flat // voc_size  # trans_indices are used to represent to different beams, values lies from k-dead_k
				word_indices = ranks_flat % voc_size    # word_indices are used to represent to different label, values lies from 0-voc_size
				costs = cand_flat[ranks_flat]
				new_hyp_samples = []
				new_hyp_scores = np.zeros(k-dead_k).astype('float32')
				new_hyp_states = []
				new_hyp_alpha_past = []

				for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
					new_hyp_samples.append(hyp_samples[ti]+[wi])    # concatenates [wi] list with list referring to beam ti 
					new_hyp_scores[idx] = copy.copy(costs[idx])
					new_hyp_states.append(copy.copy(next_state[ti]))
					new_hyp_alpha_past.append(copy.copy(next_alpha_past[ti]))

				new_live_k = 0
				hyp_samples = []
				hyp_scores = []
				hyp_states = []
				hyp_alpha_past = []

				for idx in range(len(new_hyp_samples)):
					if new_hyp_samples[idx][-1] == 0: # <eol>
						sample.append(new_hyp_samples[idx])
						sample_score.append(new_hyp_scores[idx])
						dead_k += 1
					else:
						new_live_k += 1
						hyp_samples.append(new_hyp_samples[idx])
						hyp_scores.append(new_hyp_scores[idx])
						hyp_states.append(new_hyp_states[idx])
						hyp_alpha_past.append(new_hyp_alpha_past[idx])
				hyp_scores = np.array(hyp_scores)
				live_k = new_live_k

				if new_live_k < 1:
					break
				if dead_k >= k:
					break

				next_w = np.array([w1[-1] for w1 in hyp_samples])
				next_state = np.array(hyp_states)
				next_alpha_past = np.array(hyp_alpha_past)
        
		if not stochastic:
			# dump every remaining one
			if live_k > 0:
				for idx in range(live_k):
					sample.append(hyp_samples[idx])
					sample_score.append(hyp_scores[idx])

		return sample, sample_score,sample_att



class Model():

		def build_model(self, classes): 
 
			#### encoder setup parameters ####
			dense_blocks=3
			levels_count=16
			growth=24

			#### decoder setup parameters ####
			hidden_dim=256
			word_dim=256
			dim_attend=512

			self.x = tf.placeholder(tf.float32, shape=[None, None, None])
			self.y = tf.placeholder(tf.int32, shape=[None, None])
			self.x_mask = tf.placeholder(tf.float32, shape=[None, None, None])
			self.y_mask = tf.placeholder(tf.float32, shape=[None, None])
            
			global anno, infer_y, h_pre, alpha_past, if_trainning, num_classes

			num_classes = classes
			self.lr = tf.placeholder(tf.float32, shape=())
			if_trainning = tf.placeholder(tf.bool, shape=())
			self.alpha_reg = tf.placeholder(tf.float32, shape=())
			dense_encoder = DenseEncoder(blocks=dense_blocks,level=levels_count, growth_rate=growth, training=if_trainning)
			self.annotation, self.anno_mask = dense_encoder.dense_net(self.x, self.x_mask)

			# for initilaizing validation
			anno = tf.placeholder(tf.float32, shape=[None, self.annotation.shape.as_list()[1], self.annotation.shape.as_list()[2], self.annotation.shape.as_list()[3]])
			infer_y = tf.placeholder(tf.int64, shape=(None,))
			h_pre = tf.placeholder(tf.float32, shape=[None, hidden_dim])
			alpha_past = tf.placeholder(tf.float32, shape=[None, self.annotation.shape.as_list()[1], self.annotation.shape.as_list()[2]])
	
			contextual_attention = ContextualAttention(self.annotation.shape.as_list()[3], hidden_dim, dim_attend)
	
			decoder = Decoder(hidden_dim, word_dim, contextual_attention, self.annotation.shape.as_list()[3])
	
			self.caltext = CALText(dense_encoder, contextual_attention, decoder, hidden_dim, word_dim, self.annotation.shape.as_list()[3],num_classes,if_trainning) 
	
			self.hidden_state_0 = tf.tanh(tf.tensordot(tf.reduce_mean(anno, axis=[1, 2]), self.caltext.Wa2h, axes=1) +self.caltext.ba2h)  # [batch, hidden_dim]
	  
			self.cost = self.caltext.get_cost(self.annotation, self.y, self.anno_mask, self.y_mask, self.alpha_reg)
    
		  
			alpha_c=0.5
			vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			for vv in vs:
				if not vv.name.startswith('conv2d'):
					self.cost += 1e-4 * tf.reduce_sum(tf.pow(vv, 2))
			 
			self.p, self.w, self.h, self.alpha,self.contexV= self.caltext.get_word(infer_y, h_pre, alpha_past, anno)
    
 
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				tvars= tf.trainable_variables()
				grads, _  = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),100)
				self.trainer = optimizer.apply_gradients(zip(grads, tvars))
				
		
def model_infer(model, sess, xx_pad, max_len, beam_size): 
		annot = sess.run(model.annotation, feed_dict={model.x:xx_pad, if_trainning:False})   
		h_state = sess.run(model.hidden_state_0, feed_dict={anno:annot})
		sample, score, hypalpha= model.caltext.get_sample(  model.p, model.w, model.h, model.alpha,model.contexV,annot, h_state,beam_size, max_len, False, sess, training=False)
		score = score / np.array([len(s) for s in sample])
		ss = sample[score.argmin()]
			
		return ss, hypalpha
		
def model_getcost(model, sess, batch_x, batch_y, batch_x_m, batch_y_m): 
		pprobs, annott = sess.run([model.cost, model.annotation], feed_dict={model.x:batch_x, model.y:batch_y, model.x_mask:batch_x_m, model.y_mask:batch_y_m, if_trainning:False, model.alpha_reg:1})
			
		return pprobs	
		
		
def model_train(model, sess, batch_x, batch_y, batch_x_m, batch_y_m, lrate, alpha_reg): 

		cost_i, _ = sess.run([model.cost,model.trainer],feed_dict={model.x:batch_x, model.y:batch_y, model.x_mask:batch_x_m, model.y_mask:batch_y_m, if_trainning:True, model.lr:lrate, model.alpha_reg:alpha_reg})
	
		return cost_i	