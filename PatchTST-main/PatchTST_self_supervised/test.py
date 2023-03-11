import pickle

with open('./augment.ob', 'rb') as fp:
    list_1 = pickle.load(fp)

print(len(list_1[0]))
