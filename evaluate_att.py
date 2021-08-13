'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from Dataset import load_embedding
from time import time
import Dataset
import os
import pickle
from keras.preprocessing import sequence

#from numba import jit, autojit
import linecache
# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_data = None
_K = None

def evaluate_model(model, data, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _data
    _model = model
    _data = data
    _testRatings = data.test_rating
    _testNegatives = data.negative  #49个负样本
    _K = K


    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread，
    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)
#评估函数
def eval_one_rating(idx):
    dataset = "movielen100k"
    rating = _testRatings[idx]


    u = rating[0]
    items = _testNegatives[u]
    gtItem = rating[1]
    t_u = rating[2]
    t_item = rating[3]
    items.append(gtItem)
    #print(len(items))     #  100
    # Get prediction scores
    map_item_score = {}
    #user_vectorpath = "Data/" + dataset + "_user_50.txt"
    #item_vectorpath = "Data/" + dataset + "_item_50.txt"
    #test_user_eval_pkl = "tmp/test_user_eval" + ".pkl"
    #test_item_eval_pkl = "tmp/test_item_eval" + ".pkl"
    #user_listpath = "Data/user_item_list.txt"
    #item_listpath = "Data/item_user_list.txt"
    user_long_latent = "data/ml/user_embedding.txt"
    item_long_latent = "data/ml/item_embedding.txt"
    """
    编写一个方法，能够加载当前的文件，返回一个向量的列表
    
    """
    user_embeddings = load_embedding(user_long_latent,943,10,"u")
    item_embeddings = load_embedding(item_long_latent,1152,10,"i")
    users = np.full(len(items), u, dtype = 'int32')
    item_input = []
    user_input = []
    for i in items:
        item_input.append(item_embeddings[i])
    for j in users:
        user_input.append(user_embeddings[j])
    item_input = np.array(item_input)
    user_input = np.array(user_input)
    #user当前的交互序列 100个user，100 item，对应列表长度为100的
    #if os.path.exists(test_user_eval_pkl):
        #user_seq_pos = pickle.load(open(test_user_eval_pkl, 'rb'))
        #item_seq_pos = pickle.load(open(test_item_eval_pkl, 'rb'))
    #else:
    min_len = 8
    user_seq_pos, item_seq_pos = [],[]
    for i in items:
        user_seq, item_seq = _data.getTime_seq(u,i,t_u,t_item)
        if len(user_seq) < min_len:
            #print(user_seq) 可能为空可能包括几个
            if len(user_seq) == 0:  #为空时，我们就获取这个时间点之前的序列，或者是这个item所有交互序列
                user_seq.extend([i]*10)
                #user_seq.extend(_data.getTime_seq(u,i,t,sign="user"))
            else:   #当这个时间力度中是有数据的时候我们认为这个用户的前段时间一直和item交互
                l = min_len - len(user_seq)
                for k in range(l):
                    user_seq.insert(0,user_seq[0])
        if len(item_seq) < min_len: #item的交互序列的几种处理方式
            if len(item_seq) == 0:
                item_seq.extend([u]*10)
                #item_seq.extend(_data.getTime_seq(u,i,t,sign="item"))
            else:
                l = min_len - len(item_seq)
                for j in range(l):
                    item_seq.insert(0,item_seq[0])
        user_seq_pos.append(user_seq[-min_len:])
        item_seq_pos.append(item_seq[-min_len:])

    #输入预训练好的向量

    #user_seq_pos = sequence.pad_sequences(user_seq_pos, maxlen=5)
    #item_seq_pos = sequence.pad_sequences(item_seq_pos, maxlen=5)
    #user_list = [i+1 for i in list(users)]
    #item_list = [j+1 for j in list(items)]
    #user_l,item_l维度是100*20
    #user_l = Dataset.load_long_vector(user_list, user_vectorpath)
    #item_l = Dataset.load_long_vector(item_list, item_vectorpath)
    # 根据用户列表和item列表产生对应的交互序列 100*4,
    #user_interac_list = Dataset.load_short_list(user_listpath, user_list)
    #item_interac_list = Dataset.load_short_list(item_listpath, item_list)
    #print(user_interac_list.shape)
    #测试数据完毕
    predictions = _model.predict([np.array(user_seq_pos), np.array(item_seq_pos),user_input,item_input
                                  ], batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()     #删除最后一个值
    
    # Evaluate top rank list,获得了topK个评分高的item
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    #precision = getPrecision(ranklist,gtItem)
    #print(hr,ndcg)
    return (hr, ndcg)
#例如[0，25]，存在一个itemID和negative item构建的item列表，从预测的评分中选择分值最高的k个item列表，查看25是否在列表中
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
