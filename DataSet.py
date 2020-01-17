from torch.utils.data import Dataset
import numpy
import torch


class MyDataset(Dataset):
    def __init__(self):
        self.train_data = []
        self.train_label = []
        self.name_list = []
        pt = r"D:\Codes\AIProjects\T1\test1p50.txt"
        in_file = open(pt, "r")
        txt = in_file.readlines()
        for line in txt:
            # print(line)
            name = line.split(" ")[0]
            tot = int(line.split(" ")[1])
            s = line.split(" ")[2:]
            if len(name) < 2:
                continue
            self.name_list.append(name)
            # print(name)
            # print(len(s))
            new_s = []
            for i in range(len(s)):
                # print(s[i])
                new_s.append(float(s[i]))
            # if tot < 49:
            #     x = 49 - tot
            #     for p in range(x * 133):
            #         new_s.append(0.0)

            self.train_data.append(new_s)
            # print(new_s, len(new_s))
        # print(len(self.name_list))
        IN_SIZE = 133 * 49
        self.name_set = list(set(self.name_list))
        self.name_set.sort()
        name_dict = {}
        for i in range(len(self.name_set)):
            name_dict[self.name_set[i]] = i
        set_size = len(self.name_set)
        # print(set_size)
        for name in self.name_list:
            self.train_label.append(name_dict[name])

    def __getitem__(self, index):
        return torch.from_numpy(numpy.array(self.train_data[index]).reshape((49, 133)), ), \
               self.train_label[index]

    def __len__(self):
        return len(self.train_label)
