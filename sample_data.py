import pickle
from seqloader import TEP
from torch.utils.data import random_split

Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def small_data_maker(small_data_size, sequence_length):
    train_set_all = TEP(num=Type, sequence_length=sequence_length, is_train=True)
    test_set_all = TEP(num=Type, sequence_length=sequence_length, is_train=False)
    small_test_set, _ = random_split(train_set_all, [small_data_size, len(train_set_all)-small_data_size])
    small_train_set, _ = random_split(test_set_all, [small_data_size, len(test_set_all)-small_data_size])
    with open(f'processed_data/{sequence_length}-train_set_all.p', 'wb') as f:
        pickle.dump(train_set_all, f)
    with open(f'processed_data/{sequence_length}-test_set_all.p', 'wb') as f:
        pickle.dump(test_set_all, f)
    with open(f'processed_data/{sequence_length}-train_set_small.p', 'wb') as f:
        pickle.dump(small_train_set, f)
    with open(f'processed_data/{sequence_length}-test_set_small.p', 'wb') as f:
        pickle.dump(small_test_set, f)
