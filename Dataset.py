'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import linecache
import os
import pickle
import pandas as pd
import datetime
import time
from scipy.sparse import csr_matrix
# from data-split import load_data
import operator

class DatasetPro(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        代码执行加载训练集和测试集函数load_data
        根据训练集获取每条记录当前用户以及项目的当前的序列
            读取所有的数据集文件构建一个user-item字典
            遍历训练集，为每个用户构建一个序列表
        测试集
            获取每个交互记录的
        '''
        self.dataset = "ml"
        self.moviesLen_split_data = "2002-09-28"
        self.item_delta = 30
        self.user_delta = 7
        self.rating_flag = 0
        self.is_sparse = 0
        self.user_windows_size = 4
        self.sparse_tensor = 0
        self.threshold = 10
        self.ratio = 0.8
        #self.trainMatrix = self.load_rating_file_as_matrix(path + "train_data.rating")
        #self.testRatings = self.load_rating_file_as_list(path + "test_data.rating")
        #self.testNegatives = self.load_negative_file(path + "test_negative.txt")
        self.data = self.load_data(path+"/ml/rating-delete-missing-itemid.txt")      #删除噪声数据之后得到的数据
        self.num_user = self.data.userid.unique().max() + 1       #6040
        self.num_item = self.data.itemid.unique().max() + 1     #3260
        print(self.num_user, self.num_item)     #movielen100k 943 1152
        #划分数据集
        train_file_path = "data/ml/train_data.txt"
        test_file_path = "data/ml/test_data.txt"
        #划分数据集
        #self.data_split(self.data,train_file_path,test_file_path,self.ratio)
        #加载进数据集
        self.train = self.load(train_file_path)
        self.test = self.load(test_file_path)
        self.test_rating = self.load_test_rating(self.test)  # [[u,i,t].[]...]
        #构建负样本
        self.negative = self.load_negative(self.num_item, self.data, 49)

        # print(self.data)
        # 使用的时候自动生成， 在生成负样本的过程中用到
        # self.data_matrix = self.load_data_matrix(self.data, sign="data")
        #
        # 划分数据集，得到每个用户的交互信息，不管是
        # 得到每个user以及item 在不同的时间粒度上的item 或者user
        # #self.user_dict, self.item_dict = self.getdicts()
        # #得到每个用户的交互记录字典，可以用于生成用户的交互序列，可以在用到的地方自动生成
        # #self.user_list, self.item_list = self.get_dict()        #用户与item交互序列
        # self.train, self.test = self.split_data(self.data)
        # #函数在需要的时候调用
        # #self.train_matrix = self.load_data_matrix(self.train,sign="train")
        # self.test_rating = self.load_test_rating(self.test)  # [[u,i,t].[]...]
        # #self.test_negative = self.load_test_negative(self.test_rating, 49)
        # #assert len(self.testRatings) == len(self.testNegatives)
        # self.negative = self.load_negative(self.num_item, self.data, 49)
        # #self.num_users, self.num_items = self.trainMatrix.shape
    def load(self,filename):
        file = pd.read_table(filename,names=["userid", "itemid", "rating", "timestamp",
                                      "days","item_granularity","user_granularity"])
        return pd.DataFrame(file)
    def data_split(self,rating_data,fw_train,fw_test,ratio):
        """
        1.先将一致的用户聚齐，然后根据最终count值
        2. 根据给定的比例值，计算一个确切的值，
        3.对当前用户按照时间的先后顺序进行排序
        4. 前面的一部分分配给train，后面的分配给test
        :param rating_data:
        :param fw_train:
        :param fw_test:
        :param ratio:
        :return:
        """
        train_file = open(fw_train,'w')
        test_file = open(fw_test,'w')
        userlist = list(self.data.userid.unique())
        for user in userlist:
            user_frame = self.data[self.data.userid.isin([user])]
            num = user_frame.shape[0]
            train_len = int(round(num * ratio, 0))
            user_train = user_frame.sort_values(by="timestamp")[0:train_len]
            user_test = user_frame.sort_values(by="timestamp")[train_len:]
            train_line = user_train.values.tolist()
            test_line = user_test.values.tolist()
            for line in train_line:
                train_file.write("\t".join(list(map(str,line)))+"\n")
            for li in test_line:
                test_file.write("\t".join(list(map(str,li)))+"\n")
            print(user_frame.shape[0])
        train_file.close()
        test_file.close()


    def get_dict(self):
        list_users = "tmp/" + self.dataset + "_user_intection_list" + ".pkl"
        list_items = "tmp/" + self.dataset + "_item_intection_list" + ".pkl"
        if os.path.exists(list_users) and os.path.exists(list_items):
            start = time.time()
            import gc
            gc.disable()
            user_dict = pickle.load(open(list_users, 'rb'))
            item_dict = pickle.load(open(list_items, 'rb'))
            gc.enable()
            print("load intec list cost time: %.5f " % (time.time() - start))
        else:
            print("build user-item list data...")
            user_dict, item_dict = {}, {}  # 每个user_dict = {uid:{user_granularity:dataframe(包含itemID和rating)}}
            user_windows = self.data.groupby("userid").apply(self.get_user_list, user_dict=user_dict)
            item_windows = self.data.groupby("itemid").apply(self.get_item_list, item_dict=item_dict)
            pickle.dump(user_dict, open(list_users, 'wb'), protocol=2)  # 序列化保存数据
            pickle.dump(item_dict, open(list_items, 'wb'), protocol=2)  # 序列化保存数据
        return user_dict, item_dict

    def get_user_list(self,group,user_dict):
        uid = (int(group["userid"].mode()))  # 每个group中的userid
        user_dict.setdefault(uid, [])
        user_dict[uid] = list(group.itemid)
        return len(group["itemid"].unique())
    def get_item_list(self,group,item_dict):
        itemid = (int(group["itemid"].mode()))  # 每个group中的userid
        item_dict.setdefault(itemid, [])
        item_dict[itemid] = list(group.userid)
        return len(group["userid"].unique())

    def split_data(self,data):
        train_pkl = "tmp/" + self.dataset + "_train_data"+".pkl"
        test_pkl = "tmp/" + self.dataset + "_test_data" + ".pkl"
        if os.path.exists(train_pkl) and os.path.exists(test_pkl):
            print("train-test data load over")
            return pickle.load(open(train_pkl, 'rb')), pickle.load(open(test_pkl, 'rb'))
        print("build train-test dataset")
        test_List = [0]*len(self.data.userid.unique())
        data.groupby("userid").apply(self.get_test, test_List)
        test = pd.DataFrame(test_List,
                                 columns=['userid', 'itemid', 'rating', 'timestamp', 'days', 'item_granularity',
                                          'user_granularity'])
        df1 = self.data.append(test)
        df1 = df1.append(test)
        train = df1.drop_duplicates(
            subset=['userid', 'itemid', 'rating', 'timestamp', 'days', 'item_granularity', 'user_granularity'],
            keep=False)
        #训练数据的时序
        #train = train.sort_values(by="timestamp",ascending=)
        pickle.dump(train, open(train_pkl, 'wb'), protocol=2)
        pickle.dump(test, open(test_pkl, 'wb'), protocol=2)
        return train, test
    def load_data_matrix(self,data,sign):
        test_pkl = "tmp/" + self.dataset + "_user_item_" + sign + ".pkl"
        if os.path.exists(test_pkl):
            #print(sign + "matrix load over")
            return pickle.load(open(test_pkl, 'rb'))
        mat = sp.dok_matrix((self.num_user, self.num_item), dtype=np.float32)
        for index, line in data.iterrows():
            u, i, = int(line["userid"]), int(line["itemid"])
            mat[u,i] = 1.0
        pickle.dump(mat, open(test_pkl, 'wb'), protocol=2)
        return mat
    def load_negative(self,num_item, data, num):
        """
        根据数据中每个用户交互过的item，得到user没有交互的item
        1. 构建item的集合
        2. 找出每个用户交互过的item集合
        3. 求出这两集合的差集（就是当前user未交互过的item）
        4. 随机打乱顺序并取出其中的49个作为当前用户的negative item
        :param num_item:
        :return:
        """
        test_pkl = "tmp/" + self.dataset + "_user_item_testnegative" + ".pkl"
        if os.path.exists(test_pkl):
            print("data negative load over")
            return pickle.load(open(test_pkl, 'rb'))
        print("build test negative data ...")
        user_negative_list = []
        users = list(np.arange(self.num_user))
        for u in users:
            u_df = self.data[self.data.userid.isin([u])]
            uid = self.negative_list(u_df,user_negative_list)
        pickle.dump(user_negative_list, open(test_pkl, 'wb'), protocol=2)
        return user_negative_list
    # []
    def negative_list(self,group,negative_list):
        uid = group.userid.unique()
        items = set(np.arange(self.num_item))
        item_intec = set(group.itemid)
        item_neg = list(items - item_intec)
        item_neg_list = np.random.shuffle(item_neg)
        negative_list.append(item_neg[:49])
        return uid

    def load_test_negative(self, testdata, num):    #产生user用户从未交互的交互数据，用于测试
        test_pkl = "tmp/user_item_testnegative" + ".pkl"
        if os.path.exists(test_pkl):
            print("data load over")
            return pickle.load(open(test_pkl, 'rb'))
        print("build test negative data ...")
        test_negative = []
        for line in testdata:
            negative_data = []
            u = line[0]
            for i in range(num):
                j = np.random.randint(low=0,high=self.num_item)
                while (u,j) in self.load_data_matrix(self.data, sign="data"):       #
                    j = np.random.randint(low=0,high=self.num_item)
                negative_data.append(j)
            test_negative.append(negative_data)
        pickle.dump(test_negative, open(test_pkl, 'wb'), protocol=2)
        return test_negative
    def load_test_rating(self,testdata):    #加载测试评分，指示时间的标志
        test_rat_pkl = "tmp/" + self.dataset + "_user_item_testrating" + ".pkl"
        if os.path.exists(test_rat_pkl):
            print("build test rating data over")
            return pickle.load(open(test_rat_pkl, 'rb'))
        rating_list = []
        for index, line in testdata.iterrows():     #打印前5行
            user, item, t_u,t_item= int(line["userid"]), int(line["itemid"]), line["user_granularity"],line["item_granularity"]
            rating_list.append([user,item,t_u,t_item])
        pickle.dump(rating_list, open(test_rat_pkl, 'wb'), protocol=2)
        return rating_list
    def get_test(self, group, test):
        uid = (int(group["userid"].mode()))
        test_data = group.sort_values(by="timestamp", ascending=False)[0:1]
        test[uid] = test_data.values.tolist()[0]
    def getdicts(self):
        dict_pkl = "tmp/user_item_" + self.dataset + ".pkl"
        if os.path.exists(dict_pkl):
            start = time.time()
            import gc
            gc.disable()
            [user_dict, item_dict] = pickle.load(open(dict_pkl, 'rb'))
            gc.enable()
            print("load dict cost time: %.5f " % (time.time() - start))
        else:
            print("build user-itme dict data...")
            user_dict, item_dict = {}, {}  # 每个user_dict = {uid:{user_granularity:dataframe(包含itemID和rating)}}
            user_windows = self.data.groupby("userid").apply(self.user_windows_apply, user_dict=user_dict)
            item_windows = self.data.groupby("itemid").apply(self.item_windows_apply, item_dict=item_dict)
            pickle.dump([user_dict, item_dict], open(dict_pkl, 'wb'), protocol=2)  # 序列化保存数据

        return user_dict, item_dict
    def user_windows_apply(self, group, user_dict):
        uid = (int(group["userid"].mode()))  # 每个group中的userid
        user_dict.setdefault(uid, {})
        for user_granularity in list(group.sort_values(by="timestamp")["user_granularity"].unique()):  # 按照时间粒度将数据划分成几个不同部分
            # print (group[group.user_granularity==user_granularity])

            # 如果rating_flag值为0，选出时间粒中评分不为0的item
            #user_dict[uid].setdefault(user_granularity,[])
            user_dict[uid][user_granularity] = list(group[(group.user_granularity == user_granularity)]["itemid"])
        return len(group["user_granularity"].unique())  # 返回用户粒度的数量
    def item_windows_apply(self, group, item_dict):
        itemid = (int(group["itemid"].mode()))
        # user_dict[uid]= len(group["days"].unique())
        item_dict.setdefault(itemid, {})
        for item_granularity in list(group.sort_values(by="timestamp")["item_granularity"].unique()):
            # print (group[group.user_granularity==user_granularity])

            #item_dict[itemid].setdefault(item_granularity,[])
            item_dict[itemid][item_granularity] = list(group[(group.item_granularity == item_granularity)]["userid"])
            # print (item_dict[itemid][item_granularity])
        return len(group["item_granularity"].unique())
    def load_data(self,path):
        # 创建一个文件夹，存放所有数据的序列化
        self.create_dirs("tmp")  # 创建文件夹
        dataset_pkl = "tmp/" + self.dataset +  ".pkl"
        if os.path.exists(dataset_pkl):
            print("data load over")
            return pickle.load(open(dataset_pkl, 'rb'))  # 从文件中重构python文件对象
        print("build data...")
        # 文件不存在的时候，加载文件地址
        #filename = "Data/data.rating"

        df = pd.read_table(path, sep="\t", names=["userid", "itemid", "rating", "timestamp"], engine="python")
        # 按照用户ID和itemID进行排序
        # Movielen1M 数据集时间：2000-04-25 23:05:32   ----  2003-02-28 17:49:50
        df = df.sort_values(["userid","itemid"], ascending=False)
        # 943 user
        print("there are %d users in this dataset" % (df["userid"].unique().max()))

        # split_data划分数据集应该是2017-03-04，类似的数据集
        y, m, d = (int(i) for i in self.moviesLen_split_data.split("-"))
        df["timestamp"] = [datetime.datetime.utcfromtimestamp(i).strftime("%Y-%m-%d %H:%M:%S") for i in
                           df["timestamp"].tolist()]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # 计算起止时间，数值类型是整型，最后添加距离限定时间的列
        df["days"] = (pd.to_datetime(df["timestamp"]) - pd.datetime(y, m, d)).dt.days
        # 用户的时间粒度和item的时间粒度，取整除,时间粒度包含正负粒度的
        df["item_granularity"] = df["days"] // self.item_delta  # //means floor div 每七天为一个粒度
        df["user_granularity"] = df["days"] // self.user_delta  # //means floor div
        # print userid with no rating to item
        # count_ = pd.DataFrame(df.groupby("uid").size().rename("count"))
        # print(count_.sort_values(by="count"))
        # 去除交互数据少的数据

        if self.threshold > 0:  # remove the users while the rating of them is lower than threshold
            counts_df = pd.DataFrame(df.groupby('userid').size().rename('counts'))  # 每个用户至少有20个交互数据存在
            #print(counts_df.sort_values(by="counts"))
            users = set(counts_df[counts_df.counts >= self.threshold].index)
            df = df[df.userid.isin(users)]
            count_item = pd.DataFrame(df.groupby("itemid").size().rename("counts"))
            items = set(count_item[count_item.counts >=self.threshold].index)
            df = df[df.itemid.isin(items)]

        df["rating"] = (df["rating"] > 0).astype("int")
        #       re-arrange the user and item index from zero
        df['u_original'] = df['userid'].astype('category')
        df['i_original'] = df['itemid'].astype('category')
        df['userid'] = df['u_original'].cat.codes
        df['itemid'] = df['i_original'].cat.codes
        df = df.drop('u_original', 1)
        df = df.drop('i_original', 1)
        df = df.sort_values(by="timestamp")
        print(df.userid.unique())
        # count_ = pd.DataFrame(df.groupby("uid").size().rename("count"))
        # print(count_.sort_values(by="count"))
        # 序列化对象到一个可写的文件中，以protocol的模式，0，1，2三种模式，0默认ASCALL,1旧二进制，2新二进制
        pickle.dump(df, open(dataset_pkl, 'wb'), protocol=2)
        return df
    def getTime_seq(self,user, item, t_u, t_item):
        u_seqs, i_seqs = [], []
        i_seqs.extend(list(self.data[self.data.itemid.isin([item])][
                               self.data[self.data.itemid.isin([item])].item_granularity <= t_item].userid))
        u_seqs.extend(list(self.data[self.data.userid.isin([user])][
                               self.data[self.data.userid.isin([user])].user_granularity <= t_u].itemid))
        return u_seqs, i_seqs
        """
        if sign == "item":   #对序列为空的item和user
            i_seqs.extend(list(self.data[self.data.itemid.isin([item])][self.data[self.data.itemid.isin([item])].item_granularity <= t].userid))
            return i_seqs
            #list(self.data.groupby(item)["item_granularity"])
        if sign == "user":
            u_seqs.extend(list(self.data[self.data.userid.isin([user])][self.data[self.data.userid.isin([user])].item_granularity <=t].itemid))
            return u_seqs
        else:
            for i in range(t - self.user_windows_size, t):  # 只有选中的时间窗口的粒度len(u_seqs)=4
                u_seqs.append(self.user_dict[user].get(i, None))  # 得到每个用户相对粒度的item
                i_seqs.append(self.item_dict[item].get(i, None))
            return self.getUserVector_raw(u_seqs), self.getItemVector_raw(i_seqs)
        """
    def getUserVector_raw(self, user_sets):
        u_seqs = []     #将这个时间点的item都转化为id
        for user_set in user_sets:  # user_sets = [[],[],[],[]]
            #u_seq = [0] * (self.num_item)
            #u_seq = []
            if not user_set is None:
                u_seqs.extend(user_set)
                #for index, row in user_set.iterrows():
                    #u_seq[row["itemid"]] = row["rating"]    #可以看作是当前时间，用户的隐特征
                    #u_seq.append(row["itemid"])
                #u_seqs.extend(user_set)  # [[0,0,1,0.....],[...],[],[]],形成的列表将是4*i_cnt
        #如果当前用户的4个时间粒度中交互列表的长度不够，
        #return np.array(u_seqs)
        return u_seqs
    def getItemVector_raw(self, item_sets):
        i_seqs = []
        for item_set in item_sets:
            #i_seq = [0] * (self.num_user)
            #i_seq = []
            if not item_set is None:
                i_seqs.extend(item_set)
                #for index, row in item_set.iterrows():
                    #i_seq[row["userid"]] = row["rating"]
                    #i_seq.append(row["userid"])
            #i_seqs.extend(i_seq)
        #return np.array(i_seqs)
        return i_seqs
    def getItemVector(self,item_sets):
        rows=[]
        cols=[]
        datas=[]
        for index_i,item_set in enumerate(item_sets):
            if not item_set is None:
                for index_j,row in item_set.iterrows():
                    rows.append(index_i)
                    cols.append(row["userid"])
                    datas.append(row["rating"])
        if self.sparse_tensor:
            return ( rows,cols ,datas)
        result=csr_matrix((datas, (rows, cols)), shape=(self.user_windows_size, self.num_user))
        return result
    def getUserVector(self,user_sets):
        rows=[]
        cols=[]
        datas=[]
        for index_i,user_set in enumerate(user_sets):     #4个时间粒度的数据框，使用这个函数返回索引值和每个数据框
            if not user_set is None:
                for index,row in user_set.iterrows():
                    rows.append(index_i)        #[0,1,2,3]
                    cols.append(row["itemid"])  #[]
                    datas.append(row["rating"])
        if self.sparse_tensor:     #稀疏张量，测试精确度
            return ( rows,cols ,datas)          #4个时间粒度的数据全部按照顺序放到列表中 ([0,1,2,3],[t_itemid_i,...],[rating0,1,2,3])
        return csr_matrix((datas, (rows, cols)), shape=(self.user_windows_size, self.num_item))
    def create_dirs(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat
def load_embedding(filename,num,dim,flag):
    file = open(filename,'r')
    #创建向量
    embedding = np.random.rand(num,dim)
    # 将1-943的数据转化到0-942的数组中
    id_dict = []
    for line in file:
        id = line.split("|")[0].replace(flag,"")
        # print(id)
        # id_dict.append(id)
        lines = line.split("|")[1]
        #print([float(i) for i in lines.replace("\n", "").split(" ")])
        embedding[int(id)-1] = np.array([float(i) for i in lines.replace("\n","").split(" ")])
    return embedding
def load_longshort_vector(user_list,item_list,user_vectorpath,item_vectorpath):
    user_long = []
    item_long = []
    for idx in user_list:
        user_input = get_line_context(user_vectorpath,idx)
        user_input = list(map(float,user_input.split(" ")))
        user_long.append(user_input)
    for idx in item_list:
        item_input = get_line_context(item_vectorpath,idx)
        item_input = list(map(float,item_input.split(" ")))
        item_long.append(item_input)
    return user_long, item_long
def load_long_vector(interac_list, path):
    long = []
    for idx in interac_list:
        line = get_line_context(path, idx)
        input_vector = list(map(float, line.split(" ")))
        long.append(input_vector)
    return np.array(long)
def get_line_context(user_vectorpath,idx):
    return linecache.getline(user_vectorpath,idx).strip()

def load_short_list(path, load_list):
    intec_list = []
    for idx in load_list:
        line = get_line_context(path,idx)
        line_list = list(map(int,line.split(" ")))
        intec_list.append(line_list)
    return np.array(intec_list)

if __name__ == '__main__':
    #movielen100k 的数据集
    """
    Data文件夹中创建一个movielen100k的文件夹（放评分数据集，其次放提前训练好的长期的向量）
    加载数据集，并做处理（将交互不足10个的user以及item 删掉并重新排序
    """
    data = DatasetPro('data/')
    filename = "data/ml/item_embedding.txt"
    item_embeddings,id = load_embedding(filename, 1152, 10, "i")
    print(id)
    #print(count)
    #print(data.data.itemid.max())       #0-3259
    # k = data.item_dict.keys()
    # print(3260 in k)
    #print(embedding)