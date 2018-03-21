"""All weights are normalized using user (or item) norms."""

from collections import defaultdict
from math import sqrt


class User(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = defaultdict(float)
        self.norm = 0.0

    def update_norm(self):
        self.norm = sqrt(sum([item_weight ** item_weight for item_weight in self.user_profile.values()]))


class Item(object):
    def __init__(self, item_id):
        self.item_id = item_id
        self.item_profile = defaultdict(float)
        self.norm = 0.0

    def update_norm(self):
        self.norm = sqrt(sum([user_weight ** user_weight for user_weight in self.item_profile.values()]))


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

        user.update_norm()
        item.update_norm()

    def recommend(self, user_id):
        if user_id not in self.users:
            return []

        scores = defaultdict(float)  # {item_id => relevance for the user}
        for interest_id, interest_relevance in self.users[user_id].user_profile.items():
            for neighbour_id in self.items[interest_id].item_profile:
                for candidate_id, candidate_relevance in self.users[neighbour_id].user_profile.items():
                    # Do not recommend items such that user interacted with them already.
                    if candidate_id in self.users[user_id].user_profile:
                        continue

                    scores[candidate_id] += interest_relevance / self.users[user_id].norm * \
                                            self.users[neighbour_id].user_profile[interest_id] / self.users[neighbour_id].norm * \
                                            candidate_relevance

        return sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)[:10]
