import glob
import random
import os
import sys
import numpy as np
# from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import networkx as nx
from path_extraction_ml import *


class ListDataset(Dataset):
    def __init__(self, list_path, aux_path, normalized_labels=True):
        """
		list_path：数据集的路径，这个文件是一个 txt 文件，里面每一行是一个图片的路径
		"""

        with open(list_path, "r") as file:
            self.data = file.readlines()

        self.train_data = [
            line.replace("\n", "")
            for line in self.data
        ]

        # 添加附加信息
        # Graph = nx.Graph()
        # self.Graph = add_auxiliary_into_graph(aux_path, Graph)
        # self.normalized_labels = normalized_labels
        # self.batch_count = 0
        # self.train_dict = train_dict
        # self.paths_between_pairs = {}
        # self.positive_labels = []

    def __getitem__(self, index):
        # ---------
        #  user, item
        # ---------
        line = self.train_data[index % len(self.train_data)].rstrip().split('\t')
        user = 'u' + line[0]
        item = 'i' + line[1]
        # if user not in self.train_dict:
        #     self.train_dict.update({user: [item]})
        # else:
        #     self.train_dict[user].append(item)
        # if (user, item) not in self.positive_labels:
        #     self.positive_labels.append((user, item))
        # self.Graph = add_user_movie_interaction_into_graph((user, item), self.Graph)
        # 挖掘路径
        # pos_list = dump_paths(self.Graph, (user, item), maxLen=5, sample_size=5)
        #
        # if len(pos_list) != 0:
        #     if (user, item) not in self.paths_between_pairs:
        #         self.paths_between_pairs.update(({(user, item): pos_list}))

        return user, item

    def collate_fn(self, batch):
        users, items = list(zip(*batch))
        # Remove empty placeholder targets
        # print(pos_list)
        return users, items

    def __len__(self):
        return len(self.train_data)


class TestDataProcess(Dataset):
    def __init__(self, file_path):
        """

        :param file_path: 测试集路径
        """
        with open(file_path, "r") as file:
            self.data = file.readlines()

        self.test_data = [
            line.replace("\n", "")
            for line in self.data
        ]
        # self.test_dict = {}
        print("initing 初始化成功")

    def __getitem__(self, index):
        line = self.test_data[index % len(self.test_data)].rstrip().split('\t')
        user = 'u' + line[0]
        item = 'i' + line[1]
        # if user not in self.test_dict:
        #     self.test_dict.update({user: [item]})
        # else:
        #     self.test_dict[user].append(item)
        # print(user, item)
        return user, item

    def collate_fn(self, batch):
        users, items = list(zip(*batch))
        # Remove empty placeholder targets
        # print(pos_list)
        return users, items

    def __len__(self):
        return len(self.test_data)
