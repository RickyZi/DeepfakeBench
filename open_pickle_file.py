# open a pickle file and print the content

import pickle

picke_path = './data_dict_train.pickle'

with open(picke_path, 'rb') as f:
    data_dict = pickle.load(f)

print(data_dict)

# the pickle file do in fact contain imgs path and labels as expected!