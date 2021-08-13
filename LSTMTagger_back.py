import numpy as np
import torch
import torch.autograd as autograd
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

class LSTMTagger(nn.Module):
	'''
	recurrent neural network  
	'''
	def __init__(self, node_size, input_dim, hidden_dim, out_dim, pre_embedding, \
		nonlinearity = 'relu', n_layers = 1, dropout = 0.2):
		super(LSTMTagger, self).__init__()
		self.node_size = node_size
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.out_dim = out_dim
		self.pre_embedding = pre_embedding
		self.embedding = nn.Embedding(node_size, input_dim)	#所有节点(n*input_dim)
		self.embedding.weight = nn.Parameter(pre_embedding)
		self.lstm = nn.LSTM(input_dim, hidden_dim)
		self.linear = nn.Linear(hidden_dim + self.input_dim*2, out_dim, bias=True)
		self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout)
		# self.lstm_u = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout)
		# self.lstm_i = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout)
		# self.dnn = nn.Linear(hidden_dim * 4, out_dim, bias=True)
					
	# 	每个user的前项反馈，数据输入
	def forward(self, paths_between_one_pair):
		#paths_between_one_pair_id 是输入数据
		paths_between_one_pair_id = paths_between_one_pair[0]
		user_id = paths_between_one_pair[1]
		item_id = paths_between_one_pair[2]
		#paths_between_one_pair_id=tensor([[0,1,4,3]])
		#user_id=tensor(0)
		#item_id=tensor(3)
		# user_list_id = paths_between_one_pair[1]
		# print(user_list_id)
		# u_len = len(user_list_id)
		# item_list_id = paths_between_one_pair[2]
		# print(paths_between_one_pair_id)
		sum_hidden = Variable(torch.Tensor(), requires_grad=True)
		paths_size = len(paths_between_one_pair_id)	#也就是每对user-item存在几条线路
		# user_embedding = self.embedding(user_list_id)
		# user_embedding = user_embedding.view(u_len, 1, self.input_dim)
		# print("user_embedding维度"+str(user_embedding.shape))#(4,10)
		# item_embedding = self.embedding(item_list_id)
		# item_embedding = item_embedding.view(u_len, 1, self.input_dim)
		# print(paths_size)
		for i in range(paths_size):
			path = paths_between_one_pair_id[i]    #path=tensor([ 0,  1,  4,  3], device='cuda:0')
			path_size = len(path)   #path_size=4
			# print(path_size)
			path_embedding = self.embedding(path)
			#path_embedding=
			# tensor([[0.9366, 0.3048, 0.5165, 0.8803, 0.3034, 0.4557, 0.4445,
			#		 0.4013, 0.3616, 0.8387],
			#		[0.5009, 0.4312, 0.6184, 0.6096, 0.7151, 0.9082, 0.9420,
			#		 0.3849, 1.0810, 0.9402],
			#		[0.0126, 0.3496, 0.9105, 0.8721, 0.6062, 0.8226, 0.2842,
			#		 0.5506, 0.3361, 0.4087],
			#		[0.7836, 1.1316, 1.3032, 0.9528, 0.9612, 0.5770, 0.8910,
			#		 1.3530, 0.9553, 0.5668]], device='cuda:0')
			path_embedding = path_embedding.view(path_size, 1, self.input_dim)
			if torch.cuda.is_available():
				path_embedding = path_embedding.cuda()
				# user_embedding = user_embedding.cuda()
				# item_embedding = item_embedding.cuda()
			path_out, h = self.lstm(path_embedding)
			# h=#此时的隐特征的维度(1,1,16)
			# path_out.shape=(4,1,32)
			# 返回的隐藏层的向量
			if i == 0:
					sum_hidden = h[0]
			else:		#返回的是所有路径的隐语义向量
					sum_hidden = torch.cat((sum_hidden, h[0]), 0)
		# print(sum_hidden.shape)		#2,1, pool = nn.MaxPool2d((1,16))16
		#每对交互的向量化
		user_emb = self.embedding(user_id)
		item_emb = self.embedding(item_id)
		cat = torch.cat((user_emb,item_emb),-1)  #cat  (1,20)
		cat = cat.view(1,self.input_dim * 2)
		# print(cat.shape)
		pool = nn.MaxPool2d((1,self.hidden_dim))
		# pool = nn.MaxPool2d((paths_size,1),(paths_size,1))
		# adp_pool = nn.AdaptiveAvgPool1d((1,self.hidden_dim))
		att = pool(sum_hidden)
		att = F.softmax(att, dim=0)
		# print(att.shape)

		# print(att.shape)		#2,1,1

		# max_pool = max_pool.view(paths_size,1,1)
		path_extract = torch.mul(sum_hidden,att)	#(1,1,32)

		path_emb = torch.sum(path_extract,0,True)	#(1,1,32)
		path_extract = path_emb.view(1, -1)
		# print(path_emb.shape)
		emb = torch.cat((cat,path_extract),-1)   #emb(1,1,52)
		# print(emb.shape)
		# emb = nn.Linear(self.input_dim*2+self.hidden_dim,self.hidden_dim,bias=True)(emb)
		# print(path_emb.shape)
		# print(path_extract.shape)
		# path_emb = path_extract.view(1,-1,self.hidden_dim)
		# print(path_emb.shape)
		# path_extract = F.softmax(path_extract,dim=0)
		# path_emb_pool = nn.MaxPool2d((paths_size,1),stride=(paths_size,1))
		# path_emb = path_emb_pool(path_extract)
		# print(max_pool.shape)	(1,1,16)
		# print(path_emb.shape)
		# user和item序列语义的提取
		# u_out, _ = self.lstm_u(user_embedding)
		# i_out, _ = self.lstm_i(item_embedding)
		# u_out = u_out.view(1,u_len*self.hidden_dim)	#(1,64)
		# i_out = i_out.view(1,u_len*self.hidden_dim)
		# 对用户序列的输出和项目的近期输出(4,1,16)*(4,1,16)
		# print(u_out.shape)		#(4,1,16)
		# ui = u_out.mul(i_out)
		# ui = ui.view(1,1,-1)
		# print(ui.shape)		(1,1,64)
		# ui_out = self.dnn(ui)	#(1,1,1)
		out = self.linear(emb) #（1,1,1）
		# out = out + ui_out
		out = F.sigmoid(out)
		# 两个结果的
		return out