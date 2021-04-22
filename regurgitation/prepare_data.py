import pickle
# example of training dataset
mr_data = {
    'train':[
        {
            'folder_path': './data/mr_0',
            'seg_idx': [14],
            'start': 0,
            'end': 32
        },
        {
            'folder_path': './data/mr_0',
            'seg_idx': [14, 39],
            'start': 13,
            'end': 44
        },
    ],
    'val':[
        {
            'folder_path': './data/mr_1',
            'seg_idx': [7, 21],
            'start': 0,
            'end': 30
        }
    ],
    'test':[
        {
            'folder_path': './data/mr_1',
            'seg_idx': [7, 21],
            'start': 0,
            'end': 30
        }
    ],
}

ar_data = {
    'train':[
        {
            'folder_path': './data/ar_0',
            'seg_idx': [0,10,31],
            'start': 0,
            'end': 32
        }
    ],
    'val':[
        {
            'folder_path': './data/ar_1',
            'seg_idx': [8,17],
            'start': 0,
            'end': 24
        }
    ],
    'test':[
        {
            'folder_path': './data/ar_1',
            'seg_idx': [8,17],
            'start': 0,
            'end': 24
        }
    ],
}

# generate .pkl file
with open('./mr_dataset.pkl', 'wb') as f:
    pickle.dump(mr_data, f)

with open('./ar_dataset.pkl', 'wb') as f:
    pickle.dump(ar_data, f)