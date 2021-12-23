import lmdb
import os
from tqdm import tqdm
import random
from shutil import copyfile
import argparse


def train_test_split(datasets, train_size=0.8):
    temp, train, test = dict(), dict(), dict()

    # Get the size of all datasets
    for dataset in datasets:
        env_temp = lmdb.open(dataset)
        txn_temp = env_temp.begin()
        database_temp = txn_temp.cursor()
        try:
            temp[dataset] = int(database_temp.get(b'len').decode())
        except AttributeError as e:
            print("NOPE", e)
            print(dataset)
        database_temp.close()

    N = sum(temp.values())

    # Sample from datasets to obtain train test split
    while sum(train.values()) < N * train_size:
        s = random.choice(list(temp.keys()))
        train[s] = temp[s]

        del temp[s]  # Prevent re-sampling

    test = temp  # Everything left is test set

    print("Total size", N)
    print("Train size", sum(train.values()))
    print("Test size", sum(test.values()))

    return train, test


def create_train_test(train, test, args):
    if not os.path.exists(args.out):
        print("Creating directories '{}', '/train', '/val'".format(args.out))
        os.makedirs(args.out)
        os.makedirs(args.out + '/val')
        os.makedirs(args.out + '/train')

    def helper(data, name):
        for i, dataset in enumerate(tqdm(data, desc=name)):
            new_dir = str(i).zfill(3)
            os.makedirs(args.out + '/' + name + '/' + new_dir)

            files = os.listdir(dataset)
            for f in files:
                copyfile(dataset + "/" + f,
                         args.out + '/' + name + '/' + new_dir + "/" + f)

    helper(train, 'train')
    helper(test, 'val')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_source', type=str, required=True)
    parser.add_argument('--second_source', type=str)
    parser.add_argument('--third_source', type=str)
    parser.add_argument('--train_split', type=float, default=0.81)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--skip', type=int)

    args = parser.parse_args()

    existing_dbs = [args.first_source+'/{}'.format(i) for i in os.listdir(args.first_source)]
    if args.second_source:
        temp = [args.second_source+'/{}'.format(i) for i in os.listdir(args.second_source)[args.skip:]]
        existing_dbs.extend(temp)

    if args.third_source:
        temp = [args.third_source+'/{}'.format(i) for i in os.listdir(args.third_source)[args.skip:]]
        existing_dbs.extend(temp)

    train, test = train_test_split(existing_dbs, train_size=args.train_split)
    create_train_test(train.keys(), test.keys(), args)
