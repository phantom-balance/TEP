import pickle
from seqloader import TEP
from torch.utils.data import random_split

Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
sequence_length = 5
small_data_size10 = 10
small_data_size50 = 50
small_data_size100 = 100
small_data_size200 = 200
small_data_size500 = 500
small_data_size1000 = 1000
small_data_size2000 = 2000

train_set_all = TEP(num=Type, sequence_length=sequence_length, is_train=True)
test_set_all = TEP(num=Type, sequence_length=sequence_length, is_train=False)


def small_data_maker(small_data_size, sequence_length, is_complete_data=None):
    '''
    :param small_data_size: size of the small sample data from the entire dataset
    :param sequence_length: sequence length of the data
    :param is_complete_data: if True, returns the entire dataset and labels
    :return: returns either the entire or smaller dataset {train_data, test_data} in that order
    '''

    train_set_all = TEP(num=Type, sequence_length=sequence_length, is_train=True)
    test_set_all = TEP(num=Type, sequence_length=sequence_length, is_train=False)
    small_test_set, _ = random_split(train_set_all, [small_data_size, len(train_set_all)-small_data_size])
    small_train_set, _ = random_split(test_set_all, [small_data_size, len(test_set_all)-small_data_size])
    if is_complete_data:
        train_data = train_set_all
        test_data = test_set_all
    else:
        train_data = small_train_set
        test_data = small_test_set

    return train_data, test_data


small_train_set2000, remain_train2000 = random_split(train_set_all, [small_data_size2000, len(train_set_all)-small_data_size2000])
small_test_set2000, remain_test2000 = random_split(test_set_all, [small_data_size2000, len(test_set_all)-small_data_size2000])

small_train_set1000, remain_train1000 = random_split(remain_train2000, [small_data_size1000, len(remain_train2000)-small_data_size1000])
small_test_set1000, remain_test1000 = random_split(remain_test2000, [small_data_size1000, len(remain_test2000)-small_data_size1000])

small_train_set500, remain_train500 = random_split(remain_train1000, [small_data_size500, len(remain_train1000)-small_data_size500])
small_test_set500, remain_test500 = random_split(remain_test1000, [small_data_size500, len(remain_test1000)-small_data_size500])

small_train_set200, remain_train200 = random_split(remain_train500, [small_data_size200, len(remain_train500)-small_data_size200])
small_test_set200, remain_test200 = random_split(remain_test500, [small_data_size200, len(remain_test500)-small_data_size200])

small_train_set100, remain_train100 = random_split(remain_train200, [small_data_size100, len(remain_train200)-small_data_size100])
small_test_set100, remain_test100 = random_split(remain_test200, [small_data_size100, len(remain_test200)-small_data_size100])

small_train_set50, remain_train50 = random_split(remain_train100, [small_data_size50, len(remain_train100)-small_data_size50])
small_test_set50, remain_test50 = random_split(remain_test100, [small_data_size50, len(remain_test100)-small_data_size50])

small_train_set10, remain_train10 = random_split(remain_train50, [small_data_size10, len(remain_train50)-small_data_size10])
small_test_set10, remain_test10 = random_split(remain_test50, [small_data_size10, len(remain_test50)-small_data_size10])

with open(f'sample_data/{sequence_length}train_set_all.p', 'wb') as f:
    pickle.dump(train_set_all, f)
with open(f'sample_data/{sequence_length}test_set_all.p', 'wb') as f:
    pickle.dump(test_set_all, f)

with open(f'sample_data/{sequence_length}small_test2000.p', 'wb') as f:
    pickle.dump(small_test_set2000, f)
with open(f'sample_data/{sequence_length}small_train2000.p', 'wb') as f:
    pickle.dump(small_train_set2000, f)

with open(f'sample_data/{sequence_length}small_test1000.p', 'wb') as f:
    pickle.dump(small_test_set1000, f)
with open(f'sample_data/{sequence_length}small_train1000.p', 'wb') as f:
    pickle.dump(small_train_set1000, f)

with open(f'sample_data/{sequence_length}small_test500.p', 'wb') as f:
    pickle.dump(small_test_set500, f)
with open(f'sample_data/{sequence_length}small_train500.p', 'wb') as f:
    pickle.dump(small_train_set500, f)

with open(f'sample_data/{sequence_length}small_test200.p', 'wb') as f:
    pickle.dump(small_test_set200, f)
with open(f'sample_data/{sequence_length}small_train200.p', 'wb') as f:
    pickle.dump(small_train_set200, f)

with open(f'sample_data/{sequence_length}small_test100.p', 'wb') as f:
    pickle.dump(small_test_set100, f)
with open(f'sample_data/{sequence_length}small_train100.p', 'wb') as f:
    pickle.dump(small_train_set100, f)

with open(f'sample_data/{sequence_length}small_test50.p', 'wb') as f:
    pickle.dump(small_test_set50, f)
with open(f'sample_data/{sequence_length}small_train50.p', 'wb') as f:
    pickle.dump(small_train_set50, f)

with open(f'sample_data/{sequence_length}small_test10.p', 'wb') as f:
    pickle.dump(small_test_set10, f)
with open(f'sample_data/{sequence_length}small_train10.p', 'wb') as f:
    pickle.dump(small_train_set10, f)
