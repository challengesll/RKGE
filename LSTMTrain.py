import numpy as np
import torch
import torch.autograd as autograd
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

class LSTMTrain(object):
	'''
	recurrent neural network training process
	'''
	def __init__(self, model, iteration, learning_rate, paths_between_pairs, positive_label, \
		all_variables, all_user, all_movie):
		super(LSTMTrain, self).__init__()
		self.model = model		#已经创建的模型
		self.iteration = iteration	#训练迭代的次数
		self.learning_rate = learning_rate
		self.paths_between_pairs = paths_between_pairs		#这是{(user,item):[path]}
		self.positive_label = positive_label
		self.all_variables = all_variables
		self.all_user = all_user
		self.all_movie = all_movie
		# self.user_intec_list = user_intec_list
		# self.item_intec_list = item_intec_list


	def dump_post_embedding(self):
		'''
		dump the post-train user or item embedding
		'''
		# userE = open("data/ml/user_embedding.txt","r")
		embedding_dict = {}
		node_list = self.all_user + self.all_movie
		# print("所有节点数是"+str(len(node_list)))
		for node in node_list:
			node_id = torch.LongTensor([int(self.all_variables[node])])
			# 节点的id
			# print(node_id)
			node_id = Variable(node_id)
			if torch.cuda.is_available():
				node_id = node_id.cuda()
				#ur_id = ur_id.cuda()
			#squeeze（）函数就是去掉维度为1的维度。
			node_embedding = self.model.embedding(node_id).squeeze().cpu().data.numpy()
			if node not in embedding_dict:
				#{u123:embedding},是论文中的实体向量化
				embedding_dict.update({node:node_embedding})
		#返回每个节点的embedding
		return embedding_dict
			
		
	#@property
	def train(self):
		criterion = nn.BCELoss()  #交叉熵为损失函数
		#You may also try different types of optimization methods (e.g, SGD, RMSprop, Adam, Adadelta, etc.)
		optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
		#optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
		# print(len(self.all_user),self.all_user) 输出正常 943
		# print(len(self.all_movie),self.all_movie)	1606
		for epoch in range (self.iteration):
			running_loss = 0.0
			# 每条路径,总共的交互记录数据
			data_size = len(self.paths_between_pairs)
			# print(data_size)	#72232
			label = Variable(torch.Tensor())    
			#这个是当前的（user，item）下面的path进行获取一个用户以及item的向量
			# 获取当前user-item路径，交互记录
			# counter = 0
			for pair in self.paths_between_pairs:
				# print(pair)
				# 每个（user,item）
				# user_list = self.user_intec_list[counter]
				# user_list_id = np.array([self.all_variables[x] for x in user_list])
				# user_list_id = Variable(torch.LongTensor(user_list_id))
				# item_list = self.item_intec_list[counter]
				# item_list_id = np.array([self.all_variables[i] for i in item_list])
				# item_list_id = Variable(torch.LongTensor(item_list_id))
				# counter = counter + 1
				inter_pair_id = [self.all_variables[i] for i in pair]
				user_id = np.array(inter_pair_id[0])
				item_id = np.array(inter_pair_id[1])
				user_id = Variable(torch.LongTensor(user_id))
				item_id = Variable(torch.LongTensor(item_id))
				paths_between_one_pair = self.paths_between_pairs[pair]
				paths_between_one_pair_size = len(paths_between_one_pair)#每对（user，item）总共有多少个path
				paths_between_one_pair_id = []
				#将原来的知识图谱中的user-item对的path转换为对应的ID数组
				for path in paths_between_one_pair:
					path_id = [self.all_variables[x] for x in path]#转化为ID列表
					paths_between_one_pair_id.append(path_id) 	#[[],[],[]...]

				paths_between_one_pair_id = np.array(paths_between_one_pair_id)
				paths_between_one_pair_id = Variable(torch.LongTensor(paths_between_one_pair_id))


				if torch.cuda.is_available():
					paths_between_one_pair_id = paths_between_one_pair_id.cuda()
					user_id = user_id.cuda()
					item_id =item_id.cuda()
					# user_list_id = user_list_id.cuda()
					# item_list_id = item_list_id.cuda()
				#模型输入数据，输入的可以的是一个列表，列表中包含路径，user和item的交互列表
				out = self.model([paths_between_one_pair_id,user_id,item_id])
				out = out.squeeze()
				#与真实值的对比
				if pair in self.positive_label:
					label = Variable(torch.Tensor([1]))
				else:
					label = Variable(torch.Tensor([0])) 
				#求损失
				loss = criterion(out.cpu(), label)
				running_loss += loss.item() * label.item()
				#梯度过程
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()#反向传播求梯度
			
			print('epoch['+str(epoch) + ']: loss is '+str(running_loss))

		return self.dump_post_embedding()