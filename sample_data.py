import pickle
import random
from seqloader import TEP


Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
sequence_length_list = [3, 4, 5, 6, 7, 8, 9]
small_data_size = 2500

for idx, sequence_length in enumerate(sequence_length_list):
    train_set = TEP(num=Type, sequence_length=sequence_length, is_train=True)
    test_set = TEP(num=Type, sequence_length=sequence_length, is_train=False)
    small_train_index = random.sample(range(0, len(train_set)-1),small_data_size)
    small_test_index = random.sample(range(0, len(test_set)-1),small_data_size)
    # print(small_test_index)
    # print(small_train_index)
    with open(f'processed_data/{sequence_length}-train_set_index.p', 'wb') as f:
        pickle.dump(small_train_index, f)
    with open(f'processed_data/{sequence_length}-test_set_index.p', 'wb') as f:
        pickle.dump(small_test_index, f)


# for idx, sequence_length in enumerate(sequence_length_list):
#     all_train_set = TEP(num=Type, sequence_length=sequence_length, is_train=True)
#     all_test_set = TEP(num=Type, sequence_length= sequence_length, is_train=False)
#
#     small_test_set = random_split(all_test_set, [small_data_size, len(all_test_set)-small_data_size])
#     small_train_set = random_split(all_train_set, [small_data_size, len(all_train_set)-small_data_size])
#
#     torch.save(small_test_set, f'processed_data/{sequence_length}-test_set_small.pth.tar')
#     torch.save(small_train_set, f'processed_data/{sequence_length}-train_set_small.pth.tar')
#
#
# def small_data_maker(small_data_size, sequence_length):
#     train_set_all = TEP(num=Type, sequence_length=sequence_length, is_train=True)
#     test_set_all = TEP(num=Type, sequence_length=sequence_length, is_train=False)
#     small_test_set, _ = random_split(train_set_all, [small_data_size, len(train_set_all)-small_data_size])
#     small_train_set, _ = random_split(test_set_all, [small_data_size, len(test_set_all)-small_data_size])
#     with open(f'processed_data/{sequence_length}-train_set_all.p', 'wb') as f:
#         pickle.dump(train_set_all, f)
#     with open(f'processed_data/{sequence_length}-test_set_all.p', 'wb') as f:
#         pickle.dump(test_set_all, f)
#     with open(f'processed_data/{sequence_length}-train_set_small.p', 'wb') as f:
#         pickle.dump(small_train_set, f)
#     with open(f'processed_data/{sequence_length}-test_set_small.p', 'wb') as f:
#         pickle.dump(small_test_set, f)
