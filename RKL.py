import numpy as np
import argparse
import torch
import torch.autograd as autograd
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from LSTMTagger import LSTMTagger
from LSTMTrain import LSTMTrain
from LSTMEvaluation import LSTMEvaluation
from datetime import datetime
import pandas as pd
# from path_extraction_ml import *
from path_extraction_ml import *
import networkx as nx
from negative_sample import negative_sample
from dataProcess import *


# def load_paths(data,fr_file, isPositive):


def load_paths(data, pos_list, isPositive):
    '''
	load postive or negative paths, map all nodes in paths into ids
	此时文件中的是一系列的节点
	Inputs:
		@fr_file: positive or negative paths
		@isPositive: identify fr_file is positive or negative
	'''
    # print(data)
    # print(pos_list)
    global node_count, all_variables, paths_between_pairs, positive_label, all_user, all_movie
    # 路径长度不为0的时候
    if len(pos_list) != 0:
        for line in pos_list:  # [['u729', 'i333', 'g6', 'i683']]
            for node in line:
                if node not in all_variables:
                    all_variables.update({node: node_count})
                    node_count = node_count + 1
        if isPositive:
            if data not in positive_label:
                positive_label.append(data)
        if data not in paths_between_pairs:
            paths_between_pairs.update({data: pos_list})


# else:   #当提取的路径为0时，将将原路径放置到pos_list
#     if isPositive:
#         if data not in positive_label:
#             positive_label.append(data)
#     if data not in paths_between_pairs:
#         paths_between_pairs.update({data:[list(data)]})
# 路径长度为0的时候的做法
# for line in fr_file:
#     line = line.replace('\n', '')
#     lines = line.split(',')
#     user = lines[0]
#     movie = lines[-1]
#
#     if user not in all_user:
#         all_user.append(user)
#     if movie not in all_movie:
#         all_movie.append(movie)
#
#     key = (user, movie)
#     value = []
#     path = []	#存放当前行数据的路径，作为user-movie的路径
#     #如果是正样本就将当前user-movie对保存在变量当中
#     if isPositive:
#         if key not in positive_label:
#             positive_label.append(key)
#     #添加知识图谱中的实体
#     for node in lines:
#         if node not in all_variables:
#             all_variables.update({node:node_count})
#             node_count = node_count + 1
#         path.append(node)
#
#     if key not in paths_between_pairs:
#         value.append(path)
#         paths_between_pairs.update({key:value})
#     else:
#         paths_between_pairs[key].append(path)


def load_pre_embedding(fr_pre_file, isUser):
    '''
	load pre-train-user or movie embeddings

	Inputs:
		@fr_pre_file: pre-train-user or -movie embeddings
		@isUser: identify user or movie
	'''
    global pre_embedding, all_variables
    #
    for line in fr_pre_file:
        lines = line.split('|')
        node = lines[0]
        if isUser:
            node = 'u' + node
        else:
            node = 'i' + node
        # 当前输入的节点是不是在所有的实体中，在之后进行处理和加载进来
        if node in all_variables:
            node_id = all_variables[node]
            embedding = [float(x) for x in lines[1].split()]
            embedding = np.array(embedding)
            pre_embedding[node_id] = embedding  # 将预训练的向量替换了初始化的向量


# 			{id：embedding，}包含了user-item实体的所有embedding
# 将user以及item的embedding的向量抽取出来形成一个文件
def cract_embedding(embedding_dict, fileable, data_list):
    for item in data_list:
        if item in embedding_dict:
            # embedding_dict[item]
            line = str(item) + "|" + " ".join(list(map(str, embedding_dict[item]))) + "\n"
            fileable.write(line)


def load_data(fr_file):
    '''
	load training or test data

	Input:
			@fr_rating: the user-item rating data

	Output:
			@rating_data: user-specific rating data with timestamp
	'''
    # 	data_dict = {user1:[],user2:[],...}
    global all_variables, node_count, all_user, all_movie
    data_dict = {}
    # user_intec_test = []
    # item_intec_test = []
    # test_path = {}
    for line in fr_file:
        # print(line)
        # user = "u" + str(line[1])
        # item = "i" + str(line[2])
        lines = line.replace('\n', '').split('\t')
        user = 'u' + lines[0]
        item = 'i' + lines[1]
        if user not in all_variables:
            all_variables.update({user: node_count})
            node_count = node_count + 1
        if item not in all_variables:
            all_variables.update({item: node_count})
            node_count = node_count + 1
        if user not in all_user:
            all_user.append(user)
        if item not in all_movie:
            all_movie.append(item)
        # {user:[item1,item2,item3.....]}
        # 获取user和item的序列
        # pos_test = dump_paths(Graph, (user,item), maxLen=3, sample_size=5)
        # 打印测试数据
        if user not in data_dict:
            data_dict.update({user: [item]})
        elif item not in data_dict[user]:
            data_dict[user].append(item)
    # if len(pos_test) > 0:
    #     if user not in data_dict:
    #         data_dict.update({user: [item]})
    #     elif item not in data_dict[user]:
    #         data_dict[user].append(item)
    #     if user not in test_path:
    #         test_path.update({(user,item):pos_test})
    # # print(pos_test)
    # # if len(pos_test) == 0:
    # #     pos_test = []
    # #     test_path.append(pos_test)
    #     user_list, item_list = catch_list(train_dict,train_item_dict,user,item,intec_len)
    #     user_intec_test.append(user_list)
    #     item_intec_test.append(item_list)
    # if user in train_dict:
    #     lu = len(train_dict[user])
    #     if lu >= intec_len:
    #         user_list = train_dict[user][lu - intec_len:]
    #     else:
    #         user_list = train_dict[user][0:1] * (intec_len - lu)
    #         user_list.extend(train_dict[user])
    #     user_intec_test.append(user_list)
    # if item in train_item_dict:
    #     li = len(train_item_dict[item])
    #     if li >= intec_len:
    #         item_list = train_item_dict[item][li - intec_len:]
    #     else:
    #         item_list = train_item_dict[0:1] * (intec_len - li)
    #         item_list.extend(item_list)
    #     item_intec_test.append(item_list)

    # return test_path, user_intec_list,item_intec_list, data_dict
    return data_dict


def write_results(fw_results, precision_1, precision_5, precision_10, mrr_10):
    '''
	write results into text file
	'''
    line = 'precision@1: ' + str(precision_1) + '\n' + 'precision@5: ' + str(precision_5) + '\n' \
           + 'precision@10: ' + str(precision_10) + '\n' + 'mrr: ' + str(mrr_10) + '\n'
    fw_results.write(line)


# def catch_list(train_dict, train_item_dict, user, item, intec_len):
#     user_list = []
#     item_list = []
#     if user in train_dict:
#         lu = len(train_dict[user])
#         if lu >= intec_len:
#             user_list = train_dict[user][lu - intec_len:]
#         else:
#             user_list = train_dict[user][0:1] * (intec_len - lu)
#             user_list.extend(train_dict[user])
#         # user_intec_test.append(user_list)
#     if item in train_item_dict:
#         li = len(train_item_dict[item])
#         if li >= intec_len:
#             item_list = train_item_dict[item][li - intec_len:]
#         else:
#             item_list = train_item_dict[0:1] * (intec_len - li)
#             item_list.extend(item_list)
#         # item_intec_test.append(item_list)
#     return user_list, item_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=''' Recurrent Neural Network ''')

    '''
	Parameter Settings: 
	for MovieLens in terms of [input_dim, hidden_dim, out_dim, iteration, learning_rate, optimizer] is [10, 16, 1, 5, 0.2/0.1, SGD]
	for Yelp in terms of [input_dim, hidden_dim, out_dim, iteration, 
	learning_rate, optimizer] is [20, 32, 1, 5, 0.01, SGD]
	You can change optimizer in the LSTMTrain class
	'''

    parser.add_argument('--inputdim', type=int, dest='input_dim', default=10)
    parser.add_argument('--hiddendim', type=int, dest='hidden_dim', default=32)
    parser.add_argument('--outdim', type=int, dest='out_dim', default=1)
    parser.add_argument('--iteration', type=int, dest='iteration', default=20)
    parser.add_argument('--learingrate', type=float, dest='learning_rate', default=0.01)
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=32)
    parser.add_argument('--intection_length', type=int, dest='intec_len', default=4)
    parser.add_argument('--shrink', type=float, dest='shrink', default=0.08)

    parser.add_argument('--positivepath', type=str, dest='positive_path', default='data/ml/positive-path.txt')
    parser.add_argument('--negativepath', type=str, dest='negative_path', default='data/ml/negative-path.txt')
    parser.add_argument('--pretrainuserembedding', type=str, dest='pre_train_user_embedding',
                        default='data/ml/pre-train-user-embedding.txt')
    parser.add_argument('--pretrainmovieembedding', type=str, dest='pre_train_movie_embedding',
                        default='data/ml/pre-train-item-embedding.txt')

    # parser.add_argument('--auxiliary', type=str, dest='auxiliary_file', default='data/ml/small/auxiliary-mapping.txt')
    # parser.add_argument('--train', type=str, dest='train_file', default='data/ml/small/training.txt')
    # parser.add_argument('--test', type=str, dest='test_file', default='data/ml/small/test.txt')

    parser.add_argument('--auxiliary', type=str, dest='auxiliary_file', default='data/ml/auxiliary-mapping.txt')
    parser.add_argument('--train', type=str, dest='train_file', default='data/ml/training.txt')
    parser.add_argument('--test', type=str, dest='test_file', default='data/ml/test.txt')

    parser.add_argument('--results', type=str, dest='results', default='data/ml/results.txt')
    parser.add_argument('--userembedding', type=str, dest='user_embedding', default='data/ml/user_embedding.txt')
    parser.add_argument('--itemembedding', type=str, dest='item_embedding', default='data/ml/item_embedding.txt')
    parsed_args = parser.parse_args()

    input_dim = parsed_args.input_dim  # 输入维度
    hidden_dim = parsed_args.hidden_dim  # 隐藏层维度
    out_dim = parsed_args.out_dim  # 输出维度
    iteration = parsed_args.iteration  # 迭代次数
    learning_rate = parsed_args.learning_rate  # 学习率
    intec_len = parsed_args.intec_len  # 交互序列的长度
    shrink = parsed_args.shrink

    # positive_path = parsed_args.positive_path  # 正样本路径
    # negative_path = parsed_args.negative_path
    # pre_train_user_embedding = parsed_args.pre_train_user_embedding
    # pre_train_movie_embedding = parsed_args.pre_train_movie_embedding
    train_path = parsed_args.train_file
    test_file = parsed_args.test_file
    auxiliary_file = parsed_args.auxiliary_file
    results_file = parsed_args.results

    # 获取向量的文件
    # user_embedding_file = parsed_args.user_embedding
    # item_embedding_file = parsed_args.item_embedding

    start_time = datetime.now()

    # 加载数据集
    all_user = []
    all_movie = []
    all_variables = {}
    node_count = 0
    train_dict = {}
    positive_labels = []
    paths_between_pairs = {}
    # 添加辅助信息
    Graph = nx.Graph()
    Graph = add_auxiliary_into_graph(auxiliary_file, Graph)
    dataset = ListDataset(train_path, auxiliary_file)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    # 训练集加载完成
    for batch_i, (users, items) in enumerate(dataloader):
        all_user.extend(list(users))
        all_user = list(set(all_user))
        all_movie.extend(list(items))
        all_movie = list(set(all_movie))
        # print(users)
        # print("items", items)
        # 数据封装
        for i in range(len(users)):
            user = users[i]
            item = items[i]
            Graph = add_user_movie_interaction_into_graph((user, item), Graph)
            pos_list = dump_paths(Graph, (user, item), maxLen=5, sample_size=5)
            # print(pos_list)
            if users[i] not in train_dict:
                train_dict.update({user: [item]})
            else:
                train_dict[user].append(item)
            if (user, item) not in positive_labels:
                positive_labels.append((user, item))
            if pos_list is not None:
                if len(pos_list) != 0:
                    if (user, item) not in paths_between_pairs:
                        paths_between_pairs.update(({(user, item): pos_list}))
                    for path in pos_list:
                        new_nodes = [node for node in path if node not in all_variables.keys()]
                        for node in new_nodes:
                            all_variables.update({node: node_count})
                            node_count += 1
        # print(len(users))
        # print(len(pos_lists))
        # 保存所有的变量
        new_user = [user for user in users if user not in all_variables.keys()]
        for u in new_user:
            all_variables.update({u: node_count})
            node_count += 1
        new_item = [item for item in items if item not in all_variables.keys()]
        for movie in new_item:
            all_variables.update({movie: node_count})
            node_count += 1
        print("training finished")
        break

    # 加载测试集
    test_dict = {}
    test_data = TestDataProcess(test_file)
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        # collate_fn=test_data.collate_fn
    )
    print("通过")
    for batch_n, (users, items) in enumerate(test_dataloader):
        # print(batch_n)
        # 将测试集中有的，训练集中没有用户和电影添加到所有列表中
        all_user.extend([user for user in users if user not in all_user])
        all_movie.extend([item for item in items if item not in all_movie])

        new_user = [user for user in users if user not in all_variables.keys()]
        for u in new_user:
            all_variables.update({u: node_count})
            node_count += 1
        new_item = [item for item in items if item not in all_variables.keys()]
        for movie in new_item:
            all_variables.update({movie: node_count})
            node_count += 1
        for i in range(len(users)):
            user = users[i]
            item = items[i]
            if user not in test_dict:
                test_dict.update({user: [item]})
            else:
                test_dict[user].append(item)
    # fw_results = open(results_file, 'w')
    # Graph = dataset.Graph
    # paths_between_pairs = dataset.paths_between_pairs
    # positive_label = dataset.positive_labels
    # train_dict = dataset.train_dict
    train_item_dict = {}

    #    提取知识图谱的负样本和负样本路径
    # print(train_dict)
    # print(len(all_movie))
    # print(all_movie)
    all_user_negative = negative_sample(train_dict, all_movie, shrink)
    print(all_user_negative)        # {'u196': ['<generator object <genexpr> at 0x000002174AF582B0>']}
    for user in all_user_negative:
        for item in all_user_negative[user]:
            neg_list = dump_paths(Graph, (user, item), maxLen=5, sample_size=5)
            print(neg_list)
            if neg_list is None:
                continue
            elif len(neg_list) != 0:
                for path in neg_list:
                    for node in path:
                        if node not in all_variables.keys():
                            all_variables.update({node: node_count})
                            node_count += 1
                if (user, item) not in paths_between_pairs:
                    paths_between_pairs.update({(user, item): neg_list})

    print("数据加载完毕")
    print_graph_statistic(Graph)

    print('The number of all variables is :' + str(len(all_variables)))
    end_time = datetime.now()
    duration = end_time - start_time  # 持续时间长短
    print('the duration for loading user path is ' + str(duration) + '\n')
    node_size = len(all_variables)  # 也就是知识图谱中，所有实体的大小
    # 向量化所有的实体维度大小是input_dim
    print("node_size:", node_size)
    pre_embedding = torch.rand(node_size, input_dim)  # embeddings for all nodes
    # 构建模型
    print("构建model")
    start_time = datetime.now()
    model = LSTMTagger(node_size, input_dim, hidden_dim, out_dim, pre_embedding)
    print(model)
    if torch.cuda.is_available():
        # print("cuda是可以使用的")
        model = model.cuda()
    # 模型训练，返回节点的embedding
    model_train = LSTMTrain(model, iteration, learning_rate, paths_between_pairs, positive_labels, \
                            all_variables, all_user, all_movie)
    embedding_dict = model_train.train()
    # print(len(embedding_dict.keys()))
    print('model training finished')
    end_time = datetime.now()
    duration = end_time - start_time
    print('the duration for model training is ' + str(duration) + '\n')
    # 训练完成，提取向量，并写到文件中。
    # user_file = open(user_embedding_file,'w')
    # item_file = open(item_embedding_file,'w')
    print("执行到此")
    # cract_embedding(embedding_dict,user_file,all_user)
    # cract_embedding(embedding_dict,item_file,all_movie)

    start_time = datetime.now()
    # train_dict = load_data(fr_train)
    # print(train_dict)
    # test_dict = load_data(fr_test)
    print("数据加载完毕")
    # train_dict = load_data(fr_train)
    # test_dict = load_data(fr_test)
    model_evaluation = LSTMEvaluation(model, embedding_dict, all_movie, train_dict, train_item_dict,
                                      test_data.test_dict, Graph, all_variables)

    # model_evaluation = LSTMEvaluation(embedding_dict, all_movie, train_dict, test_dict)
    top_score_dict = model_evaluation.calculate_ranking_score()
    precision_1, _ = model_evaluation.calculate_results(top_score_dict, 1)
    precision_5, _ = model_evaluation.calculate_results(top_score_dict, 5)
    precision_10, mrr_10 = model_evaluation.calculate_results(top_score_dict, 10)
    precision_20, _ = model_evaluation.calculate_results(top_score_dict, 20)
    end_time = datetime.now()
    duration = end_time - start_time
    print('the duration for model evaluation is ' + str(duration) + '\n')

    # write_results(fw_results, precision_1, precision_5, precision_10, mrr_10)

    end_time = datetime.now()
    duration = end_time - start_time
    print('the duration for loading item embedding is ' + str(duration) + '\n')

    # fr_test.close()
    fr_auxiliary.close()
