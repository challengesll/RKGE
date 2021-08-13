# -*- coding: UTF-8 -*-

import numpy as np
import os
import pickle
import theano
import theano.tensor as T
import keras
from keras import backend as K
from  keras import initializers
from keras.preprocessing import sequence
from keras.regularizers import l1, l2, l1_l2
from keras.layers import *
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape,  Flatten, Dropout, LSTM
from keras.layers.merge import concatenate, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate_att import evaluate_model
from keras.layers.core import Reshape
import Dataset
from Dataset import DatasetPro,load_embedding
from time import time
import sys
#import GMF, MLP
import argparse
import linecache
from self_attention0 import *
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=150,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[100,64,32,16]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=2,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()
"""
def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)
"""
def get_model(num_users, num_items, dim=10, layers=[64,32,16,8], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    #user_long = Input(shape=(50,), dtype="float32", name="user_long")
    #item_long = Input(shape=(50,), dtype="float32", name="item_long")
    #user和item的短期向量 每个时间粒度中交互序列的one-hot值，然后我们再对输入的数据进行稠密化
    user_short = Input(shape=(8,), dtype="int32", name="user_short")
    item_short = Input(shape=(8,), dtype="int32", name="item_short")
    user_input = Input(shape=(10,), dtype='float32', name = 'user_input')
    item_input = Input(shape=(10,), dtype='float32', name = 'item_input')
    Embedding_User_short = Embedding(input_dim=num_users, output_dim=dim,name='user_embedding')
    Embedding_Item_short = Embedding(input_dim=num_items, output_dim=dim,name='item_embedding')
    #users和item的embedding
    #Embedding_User = Embedding(input_dim=num_users, output_dim=dim, name='user_embedding_user')
    #Embedding_Item = Embedding(input_dim=num_items, output_dim=dim, name='item_embedding_item')
    user_s = Embedding_Item_short(user_short)  # batch_size*length*dim
    item_s = Embedding_User_short(item_short)
    #输入的user，item向量化
    #user_latent = Embedding_User_short(user_input)
    #item_latent = Embedding_Item_short(item_input)  #batch_size*dim

    # mf_Embedding_User = Embedding(input_dim=num_users, output_dim=dim, name="mf_embedding_user",
    #                                 input_length=1)
    # mf_Embedding_Item = Embedding(input_dim=num_items, output_dim=dim, name='mf_embedding_item',
    #                                 input_length=1)
    # mf_user_latent = Flatten()(mf_Embedding_User(user_input))
    # mf_item_latent = Flatten()(mf_Embedding_Item(item_input))

    #注意力机制和item_short vector按位相乘

    user_short_latent = LSTM(50, activation="relu")(user_s)
    item_short_latent = LSTM(50, activation="relu")(item_s)
    # user_position = Position_Embedding()(user_s)
    # item_position = Position_Embedding()(item_s)
    att_user_vector = Attention(5, 10)([user_s, user_s, user_s])
    att_item_vector = Attention(5, 10)([item_s, item_s, item_s])



    mf_vector = multiply([user_input, item_input])      #mf 100
    #mf_vector = Dense(50)(mf_vector)
    #lstm获取的特征向量
    user_short_latent = concatenate([user_short_latent,att_user_vector],axis=-1)
    item_short_latent = concatenate([item_short_latent, att_item_vector], axis=-1)
    short_pre = multiply([user_short_latent, item_short_latent])
    short_pre = Dense(10)(short_pre)
    long_short_pre = Add()([mf_vector, short_pre])
    #long_short_pre = Dense(32,activation="relu")(long_short_pre)
    #long_short_pre = Dropout(0.1)(long_short_pre)
    #pre_vector = concatenate([long_short_pre,mf_vector], axis=-1)
    # 预测
    #prediction0 = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name="prediction0")(pre_vector)
    prediction = Dense(1, activation='relu', kernel_initializer='lecun_uniform', name="prediction")(long_short_pre)
    model = Model(inputs=[user_short, item_short,user_input,item_input],
                  outputs=prediction)
    model.summary()

    return model
#获取user和item的静态向量

#获取训练数据列表的
def get_train_instances(data_rat, num_negatives):
    start = time()
    user_input, item_input, labels,timeline_u,timeline_item= [],[],[],[],[]
    #num_users = train.shape[0]
    train_data = data_rat.train
    train_matrix = data_rat.load_data_matrix(train_data, sign="train")
    for index, line in train_data.iterrows():
        # positive instance
        u = line["userid"]
        user_input.append(u)
        item_input.append(line["itemid"])
        labels.append(1)
        t_u = line["user_granularity"]
        t_item = line["item_granularity"]
        timeline_u.append(t_u)
        timeline_item.append(t_item)
        # negative instances    4
        for t in range(num_negatives):
            j = np.random.randint(low=0,high=num_items)
            while (u,j) in train_matrix:
                j = np.random.randint(low=0,high=num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
            timeline_u.append(t_u)
            timeline_item.append(t_item)
    #print("cost time:",time()-start)
    return user_input, item_input, labels, timeline_u, timeline_item

if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
            
    topK = 10
    evaluation_threads = 1#mp.cpu_count()
    print("NeuMF arguments: %s " %(args))
    # model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.h5' %(args.dataset, mf_dim, args.layers, time())

    # Loading data
    t1 = time()
    dataset = DatasetPro(args.path)
    #train_data, test_data = dataset.train, dataset.test
    #train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = dataset.num_user, dataset.num_item
    #user_vectorpath = "Data/movielen100k_user_50.txt"
    #item_vectorpath = "Data/movielen100k_item_50.txt"
    #user_listpath = "Data/user_item_list.txt"
    #item_listpath = "Data/item_user_list.txt"

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, len(dataset.train), len(dataset.test)))
    
    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    

    # Init performance
    """"""
    (hits, ndcgs) = evaluate_model(model, dataset, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))

    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    #if args.out > 0:
        #model.save_weights(model_out_file, overwrite=True)
    user_long_latent = "data/ml/user_embedding.txt"
    item_long_latent = "data/ml/item_embedding.txt"
    user_embeddings = load_embedding(user_long_latent, 943, 10, "u")
    item_embeddings = load_embedding(item_long_latent, 1152, 10, "i")
    # Training model
    for epoch in range(num_epochs):

        t1 = time()
        # Generate training instances,生成用户列表，item列表（包括负样本），评分数据
        user_input, item_input, labels, timeline_u, timeline_item = get_train_instances(dataset, num_negatives)
        #print("build data down")
        #对每个user和item加载相应的向量

        """
        编写一个方法，能够加载当前的文件，返回一个向量的列表

        """

        #users = np.full(len(items), u, dtype='int32')
        item_input_embedding = []
        user_input_embedding = []
        for i in item_input:
            item_input_embedding.append(item_embeddings[i])
        for j in user_input:
            user_input_embedding.append(user_embeddings[j])
        item_input_embedding = np.array(item_input_embedding)
        user_input_embedding = np.array(user_input_embedding)
        print("向量加载完毕")
        # user_list = [i+1 for i in user_input]
        # item_list = [j+1 for j in item_input]

        #加载静态的向量,返回列表
        #user_l = Dataset.load_long_vector(user_list, user_vectorpath)
        #item_l = Dataset.load_long_vector(item_list, item_vectorpath)
        #根据用户列表和item列表产生对应的交互序列
        #直接加载或者计算
        min_len = 8

        user_intec_seq, item_intec_seq = [], []
        for i in range(len(user_input)):
            u = user_input[i]
            item = item_input[i]
            t_u = timeline_u[i]
            t_item = timeline_item[i]
            user_seq,item_seq = dataset.getTime_seq(u,item,t_u,t_item)
            if len(user_seq) < min_len:
                #print(user_seq) 可能为空可能包括几个
                if len(user_seq) == 0:  #为空时，我们就获取这个时间点之前的序列，或者是这个item所有交互序列
                    #当前序列长度为0时
                    user_seq.extend([item]*10)
                    #user_seq.extend(dataset.getTime_seq(u,item,t,sign="user"))
                else:   #当这个时间力度中是有数据的时候我们认为这个用户的前段时间一直和item交互
                    l = min_len - len(user_seq)
                    for k in range(l):
                        user_seq.insert(0,user_seq[0])
            if len(item_seq) < min_len: #item的交互序列的几种处理方式
                if len(item_seq) == 0:
                    item_seq.extend([u]*10)
                    #item_seq.extend(dataset.getTime_seq(u,item,t,sign="item"))
                else:
                    l = min_len - len(item_seq)
                    for j in range(l):
                        item_seq.insert(0,item_seq[0])
            user_intec_seq.append(user_seq[-min_len:])
            item_intec_seq.append(item_seq[-min_len:])
        print("序列获取完毕")
        #user_intec_seq = sequence.pad_sequences(user_intec_seq, maxlen=5)
        #item_intec_seq = sequence.pad_sequences(item_intec_seq, maxlen=5)
        #print("build train done,spend time:[%.1f s]"%(time() - t1))

        #t1 = time()
        #user_interac_list = Dataset.load_short_list(user_listpath, user_list)
        #item_interac_list = Dataset.load_short_list(item_listpath, item_list)
        # Training,输入数据,输入的数据包括静态的向量，用户的交互序列
        hist = model.fit([np.array(user_intec_seq), np.array(item_intec_seq), user_input_embedding,item_input_embedding], np.array(labels),
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=False)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, dataset, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best NeuMF model is saved to %s" %(model_out_file))
