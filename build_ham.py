import glob
import numpy as np

np.set_printoptions(threshold=np.nan)
dictionary = []
labels = {}

with open('./parsed/dictionary', 'r') as file:
    for line in file:
        dictionary = line.split(' ')

with open('./trec06p-cs280/labels', 'r') as file:
    for line in file:
        (label, path) = line.strip().split(' ')
        (folder, file) = path.split('/')[-2:]
        if folder not in labels:
            labels[folder] = {}
        labels[folder][file] = label

ham = None
for path in sorted(glob.glob('./parsed/training/*/*')):
    (folder, file) = path.split('/')[-2:]
    if labels[folder][file] == 'spam':
        continue

    print('processing {}'.format(path))

    with open(path, 'r') as file:
        for line in file:
            content = line.split(' ')
            document = np.array([int(w in content) for w in dictionary])

    if ham is None:
        ham = np.array(document)[None].T
    else:
        ham = np.append(
            ham,
            np.array(document)[None].T,
            axis=1
        )

np.save('./parsed/ham_matrix.npy', ham)
