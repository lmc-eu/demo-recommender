"""Recommender detect and use user neighbours."""

import logging

from collections import defaultdict
from math import sqrt


class User(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = defaultdict(float)
        self.neighbours = []
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

    @staticmethod
    def get_similarity(profile_a, norm_a, profile_b, norm_b):
        similarity = 0.0
        for i in profile_a:
            similarity += profile_a[i] * profile_b[i]
        similarity /= norm_a * norm_b
        return similarity

    def get_neighbours_candidates(self, user_profile):
        neighbour_candidates = defaultdict(int)
        for item_id in user_profile:
            for user_id in self.items[item_id].item_profile:
                neighbour_candidates[user_id] += 1

        return sorted(neighbour_candidates.keys(), key=lambda user_id: neighbour_candidates[user_id], reverse=True)[:1000]

    def detect_user_neighbours(self):
        for n_user, user_id_a in enumerate(self.users):
            if (n_user % 100) == 0:
                logging.info('Detected neighbours for %d (%d%%) users', n_user, n_user * 100 / len(self.users))

            user_profile_a = self.users[user_id_a].user_profile
            user_norm_a = self.users[user_id_a].norm
            neighbours_scored = dict()  # {neighbour_id => similarity}

            for user_id_b in self.get_neighbours_candidates(user_profile_a):
                if user_id_b == user_id_a:
                    continue

                user_profile_b = self.users[user_id_b].user_profile
                user_norm_b = self.users[user_id_b].norm

                similarity = self.get_similarity(user_profile_a, user_norm_a, user_profile_b, user_norm_b)
                neighbours_scored[user_id_b] = similarity

            sorted_neighbours_ids = sorted(neighbours_scored.keys(), key=lambda neighbour_id: neighbours_scored[neighbour_id], reverse=True)[:50]
            self.users[user_id_a].neighbours = [{'neighbour_id': neighbour_id, 'similarity': neighbours_scored[neighbour_id]} for neighbour_id in sorted_neighbours_ids]

    def recommend(self, user_id):
        if user_id not in self.users:
            return []

        scores = defaultdict(float)
        user_node = self.users[user_id]
        for neighbour in user_node.neighbours:
            neighbour_id = neighbour['neighbour_id']
            neighbour_node = self.users[neighbour_id]
            similarity = neighbour['similarity']

            for neighbour_item_id, neighbour_item_weight in neighbour_node.user_profile.items():
                if neighbour_item_id in user_node.user_profile:
                    continue

                scores[neighbour_item_id] += similarity * neighbour_item_weight

        return sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)[:10]
