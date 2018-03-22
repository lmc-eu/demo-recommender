"""Recommender detect and use user neighbours."""

import logging

from collections import defaultdict
from math import sqrt


class User(object):
    """ A data class for the user representation.
    """
    def __init__(self, user_id):
        self.user_id = user_id  # A string user identification.
        self.user_profile = defaultdict(float)  # A dict {item_id => interaction weight}
        self.neighbours = []  # A list of dict, one dict present one user neighbour and contains the keys neighbour_id and similarity.
        self.norm = 0.0   # A user norm (i.e. an euclidean norm of weights from the user profile).

    def update_norm(self):
        """ Calculate a user's norm.
        """
        self.norm = sqrt(sum([item_weight ** item_weight for item_weight in self.user_profile.values()]))


class Item(object):
    """ A data class for the item representation.
    """
    def __init__(self, item_id):
        self.item_id = item_id  # A string identification.
        self.item_profile = defaultdict(float)  # A dict {user_id => interaction weight}
        self.norm = 0.0  # A item's norm.

    def update_norm(self):
        """ Calculate the item's norm.
        """
        self.norm = sqrt(sum([user_weight ** user_weight for user_weight in self.item_profile.values()]))


class Recommender(object):
    """
    Recommend items for users using the paths (user, neighbour, item)
    This class represent the user-item network and detect user neighbours, i.e. shortcuts in the network.

    """
    def __init__(self):
        self.users = dict()  # A dict {user_id => User}
        self.items = dict()  # A dict {item_id => Item}

    def put_interaction(self, user_id, item_id, weight):
        """ Add a new edge in the network.
        """
        # Obtain the user if already exists.
        if user_id in self.users:
            user = self.users[user_id]
        # ...or create a new one.
        else:
            user = self.users[user_id] = User(user_id)

        # Obtain the item if already exists.
        if item_id in self.items:
            item = self.items[item_id]
        # ...or create a new one.
        else:
            item = self.items[item_id] = Item(item_id)

        # Insert a new interest into the user/item profile.
        # If there is the same interests, use the higher weight.
        # Here you can try different methods how to deal with repeating interactions (e.g. sum them)
        user.user_profile[item_id] = max(weight, user.user_profile[item_id])
        item.item_profile[user_id] = max(weight, item.item_profile[user_id])

    @staticmethod
    def get_similarity(profile_a, norm_a, profile_b, norm_b):
        """ Calculate the (cosine) similarity between two user profiles.
        """
        similarity = 0.0
        for i in profile_a:
            similarity += profile_a[i] * profile_b[i]
        similarity /= norm_a * norm_b
        return similarity

    def get_neighbours_candidates(self, user_profile):
        """
        A heuristic to preselect a subset from the users neighbours candidates.
        The 1000 most relevant neighbour candidates are selected for each user, based on the (normalized) number of common items.
        """
        neighbour_candidates = defaultdict(int)
        for item_id in user_profile:
            for user_id in self.items[item_id].item_profile:
                neighbour_candidates[user_id] += 1 / self.users[user_id].norm

        return sorted(neighbour_candidates.keys(), key=lambda user_id: neighbour_candidates[user_id], reverse=True)[:1000]

    def detect_user_neighbours(self):
        """
        A simple method for neighbours detection.
        For each user obtain a set of 1000 neighbour candidates. For each of them calculate the user-user similarity
        and choose the 50 most important neighbours.
        """
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
        """
        Recommend top 10 most important items.
        Recommendations are provided from all items on the path (user-neighbour-item).

        """
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
