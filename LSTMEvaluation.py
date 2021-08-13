import heapq
import numpy as np

import math
from path_extraction_ml import *
# from recurrent_knowledge_learning import catch_list
from path_extraction_ml import catch_list
from torch.autograd import Variable
import torch

class LSTMEvaluation(object):
	'''
	recurrent neural network evaluation
	'''
	def __init__(self, model,embedding_dict, all_movie, train_dict, train_item_dict,
				 test_dict,graph, all_variables,intec_len = 4):
		super(LSTMEvaluation, self).__init__()
		self.model = model
		self.embedding_dict = embedding_dict
		self.all_movie = all_movie
		self.train_dict = train_dict
		self.train_item_dict = train_item_dict
		self.test_dict = test_dict
		# self.test_path = test_path
		# self.user_intec_test = user_intec_test
		# self.item_intec_test = item_intec_test
		self.graph = graph
		self.all_variables = all_variables
		self.intec_len = intec_len
		self.mrr = 0.0


	def calculate_ranking_score(self):
		'''
		calculate ranking score of unrated movies for each user
		可以直接使用神经网络的权值来计算得到的结果
		步骤：
		1.提取测试数据中每个用户和项目的序列
		2.遍历测试数据中的每个用户，然后输入此时的用户序列和项目数列以及此时提取的路径？
		'''
		score_dict = {}
		top_score_dict = {}

		for user in self.test_dict:
			if user in self.embedding_dict and user in self.train_dict:
				for movie in self.all_movie:
					if movie not in self.train_dict[user] and movie in self.embedding_dict:
						"""
						添加测试代码
						自动的获取交互序列
						遍历每个获取的交互路径for pair in test_dict:
							path_between_pair = test_dict[dict]
							user_list_test = user_list[counter]
							item_list_test = item_list[counter]
							model.train()
							需要挖掘不同的
						"""
						# pos_list = dump_paths(self.graph, (user, movie), maxLen=3, sample_size=5)
						# if len(pos_list) != 0:
						# # 	user movie 之间的路径
						# # 	user_list, item_list = catch_list(self.train_dict,self.train_item_dict,user,movie,self.intec_len)
						# # 	user_list_id = np.array([self.all_variables[x] for x in user_list])
						# # 	user_list_id = Variable(torch.LongTensor(user_list_id))
						# # 	item_list_id = np.array([self.all_variables[i] for i in item_list])
						# # 	item_list_id = Variable(torch.LongTensor(item_list_id))
						# 	# paths_between_one_pair = self.paths_between_pairs[pair]
						# 	paths_between_one_pair_size = len(pos_list)  # 每对（user，item）总共有多少个path
						# 	paths_between_one_pair_id = []
						# 	# 将原来的知识图谱中的user-item对的path转换为对应的ID数组
						# 	for path in pos_list:
						# 		path_id = [self.all_variables[x] for x in path]  # 转化为ID列表
						# 		paths_between_one_pair_id.append(path_id)  # [[],[],[]...]
                        #
						# 	paths_between_one_pair_id = np.array(paths_between_one_pair_id)
						# 	paths_between_one_pair_id = Variable(torch.LongTensor(paths_between_one_pair_id))
                        #
						# 	if torch.cuda.is_available():
						# 		paths_between_one_pair_id = paths_between_one_pair_id.cuda()
						# 		# user_list_id = user_list_id.cuda()
						# 		# item_list_id = item_list_id.cuda()
						# 	# 模型输入数据，输入的可以的是一个列表，列表中包含路径，user和item的交互列表
						# 	out = self.model([paths_between_one_pair_id])
						# 	out = out.squeeze()
						# 	score = float(out.cpu().data.numpy())
						# 	if user not in score_dict:
						# 		score_dict.update({user:{movie:score}})
						# 	else:
						# 		score_dict[user].update({movie:score})
						embedding_user = self.embedding_dict[user]
						embedding_movie = self.embedding_dict[movie]
						score = np.dot(embedding_user, embedding_movie)
						if user not in score_dict:
							score_dict.update({user:{movie:score}})
						else:
							score_dict[user].update({movie:score})

				#rank score in a descending order
				if user in score_dict and len(score_dict[user]) > 1:
					item_score_list = score_dict[user]		#{movie:score,...}
					k = min(len(item_score_list), 20) # k=15 to speed up the ranking process, we only find the top 15 movies
					top_item_list = heapq.nlargest(k, item_score_list, key=item_score_list.get)
					top_score_dict.update({user:top_item_list})
		# print(top_score_dict)
		return top_score_dict

	def calculate_results(self, top_score_dict, k):
		'''
		calculate the final results: pre@k and mrr
		'''
		precision = 0.0
		isMrr = False
		if k == 10: isMrr = True
		# print(top_score_dict)
		user_size = 0
		hits = []
		ndcgs = []
		"""
		对于每个用户的推荐列表
		"""
		for user in self.test_dict:
			if user in top_score_dict:
				user_size = user_size + 1
				# 推荐列表
				candidate_item = top_score_dict[user]		#{movie:score,movie1:score2..}
				candidate_size = len(candidate_item)		#候选集合的大小
				hit = 0
				count = 0
				ndc = 0
				min_len = min(candidate_size, k)
				for i in range(min_len):
					if candidate_item[i] in self.test_dict[user]:
						count += 1
						hit += 1
						# hits.append(1)
						# hit = hit + 1
						ndc += math.log(2) / math.log(i+2)
						if isMrr: self.mrr += float(1/(i+1))
					# else:
					# 	hits.append(0)
				if count != 0:
					ndcgs.append(ndc / count)
					hits.append(hit/count)
				else:
					ndcgs.append(0)
				hit_ratio = float(hit / min_len)
				# hits.append(hit_ratio)
				precision += hit_ratio

		precision = precision / user_size 
		print ('precision@' + str(k) + ' is: ' + str(precision))
		print("hit_ratio@" + str(k) + "is: " + str(np.array(hits).sum() / user_size))
		print("ndcg@" + str(k) + "is: " + str(np.array(ndcgs).sum() / user_size))

		if isMrr:
			self.mrr = self.mrr / user_size
			print ('mrr@' + str(k) +' is: ' + str(self.mrr))

		return precision, self.mrr