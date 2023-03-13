import pickle
import json


f = open('char_dict', 'rb')
dic = pickle.load(f)
with open('char_dict.json', 'w') as f:
    json.dump(dic, f)
