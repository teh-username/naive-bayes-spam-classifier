import glob

dictionary = set()
for path in sorted(glob.glob('./parsed/training/*/*')):
    print('processing {}'.format(path))
    with open(path, 'r') as data:
        for line in data:
            dictionary.update(line.split(' '))

with open('./parsed/dictionary', 'w') as file:
    file.write(' '.join(dictionary))
