#!/usr/bin/env python
# coding=utf-8

import os
import sys
import zipfile

if os.path.exists('train.txt'):
    print('Tokenized enwik8 already exists - skipping processing')
    sys.exit()

data = zipfile.ZipFile('enwik8.zip').read('enwik8')

print('Length of enwik8: {}'.format(len(data)))

num_test_chars = 5000000

train_data = data[: -2 * num_test_chars]
valid_data = data[-2 * num_test_chars: -num_test_chars]
test_data = data[-num_test_chars:]

for fn, part in [('train.txt', train_data), ('valid.txt', valid_data), ('test.txt', test_data)]:
    print('{} will have {} bytes'.format(fn, len(part)))
    print('- Decoding and writing text...')
    # Write decoded text (not integer values)
    part_str = part.decode('utf-8', errors='replace')
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(part_str)
    print('- Writing raw bytes...')
    with open(fn + '.raw', 'wb') as f:
        f.write(part)

# for fn, part in [('train.txt', train_data), ('valid.txt', valid_data), ('test.txt', test_data)]:
#     print('{} will have {} bytes'.format(fn, len(part)))
#     print('- Tokenizing...')
#     part_str = ' '.join([str(c) if c != ord('\n') else '\n' for c in part])
#     print('- Writing...')
#     f = open(fn, 'w').write(part_str)
#     f = open(fn + '.raw', 'wb').write(part)