import glob
import os
import re

training_set = '070'
regex = re.compile(r'^[a-zA-Z]+[,.]?$')
destination_path = "./parsed/{}/{}/{}"

for path in sorted(glob.glob('./trec06p-cs280/data/*/*')):
    print('processing {}'.format(path))
    parsed = []
    with open(path, 'r', encoding='latin-1') as data:
        for line in data:
            words = filter(regex.search, line.strip().split(' '))
            for word in words:
                if ',' in word or '.' in word:
                    word = word[:-1]
                parsed.append(word)

    (folder, file) = path.split('/')[-2:]
    dest_path = destination_path.format(
        'training' if folder <= training_set else 'test',
        folder,
        file
    )

    if not os.path.exists(os.path.dirname(dest_path)):
        os.makedirs(os.path.dirname(dest_path))

    with open(dest_path, 'w') as file:
        file.write(' '.join(parsed))
