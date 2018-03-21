"""The same as the baseline, but items already rated by users are excluded from recommendations."""

from collections import defaultdict


class User(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = defaultdict(float)


class Item(object):
    def __init__(self, item_id):
        self.item_id = item_id
        self.item_profile = defaultdict(float)


class Recommender(object):
    def __init__(self):
        self.users = dict()
        self.items = dict()

    def put_interaction(self, user_id, item_id, weight):
        if user_id in self.users:
            user = self.users[user_id]
        else:
            user = self.users[user_id] = User(user_id)

        if item_id in self.items:
            item = self.items[item_id]
        else:
            item = self.items[item_id] = Item(item_id)

        user.user_profile[item_id] = max(weight, user.user_profile[item_id])
        item.item_profile[user_id] = max(weight, item.item_profile[user_id])

    def recommend(self, user_id):
        if user_id not in self.users:
            return []

        scores = defaultdict(float)
        user_node = self.users[user_id]
        for user_item_id, user_item_weight in user_node.user_profile.items():
            user_item_node = self.items[user_item_id]
            for neighbour_id, neighbour_weight in user_item_node.item_profile.items():
                neighbour_node = self.users[neighbour_id]
                for neighbour_item_id, neighbour_item_weight in neighbour_node.user_profile.items():
                    if neighbour_item_id in user_node.user_profile:
                        continue

                    scores[neighbour_item_id] += user_item_weight * neighbour_weight * neighbour_item_weight

        return sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)[:10]
