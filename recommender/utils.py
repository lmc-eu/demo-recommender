import logging

from collections import defaultdict
from sklearn.model_selection import train_test_split


def split_dataset(dataset):
    y = [interaction['user_id'] for interaction in dataset]
    train, test, _, _ = train_test_split(dataset, y, test_size=0.1, random_state=42)
    return train, test


def check_dataset(train_dataset, test_dataset):
    test_ratings = defaultdict(set)
    for interaction in test_dataset:
        user_id = interaction['user_id']
        item_id = interaction['item_id']
        test_ratings[user_id].add(item_id)

    train_ratings = defaultdict(set)
    for interaction in train_dataset:
        user_id = interaction['user_id']
        item_id = interaction['item_id']
        train_ratings[user_id].add(item_id)

    n_missing = 0
    for user_id in test_ratings.keys():
        if user_id not in train_ratings:
            n_missing += 1

    logging.info('Number of users that appear only in test data = %d', n_missing)


def evaluate(recommender, test_dataset):
    users_ratings = defaultdict(set)
    for interaction in test_dataset:
        user_id = interaction['user_id']
        item_id = interaction['item_id']
        users_ratings[user_id].add(item_id)

    correctly_recommended = 0
    total_recommended = 0

    for n_user, (user_id, user_ratings) in enumerate(users_ratings.items()):
        if (n_user % 100) == 0:
            logging.info('Evaluate %d users (%d%%) [%d]', n_user, n_user * 100 / len(users_ratings), correctly_recommended)

        recommended_items = set(recommender.recommend(user_id))
        correctly_recommended += len(users_ratings[user_id] & recommended_items)
        total_recommended += len(users_ratings[user_id])

    logging.info('Correctly recommended = %d', correctly_recommended)
    logging.info('Total recommended = %d', total_recommended)
    return correctly_recommended / total_recommended
