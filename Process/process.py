import os
from Process.dataset import GraphDataset, test_GraphDataset

cwd=os.getcwd()


def loadData(dataname, fold_x_train,fold_x_test,droprate):
    print("loading train set", )
    traindata_list = GraphDataset(fold_x_train, droprate=droprate)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = test_GraphDataset(fold_x_test, droprate=0)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list
