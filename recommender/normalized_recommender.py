"""All weights are normalized using user (or item) norms."""

from collections import defaultdict
from math import sqrt


class User(object):
    """ Data class that represents one user in the recommender network.
    """
    def __init__(self, user_id):
        self.user_id = user_id  # String identifier of the user.
        self.user_profile = defaultdict(float)  # A dict with users interests (interactions), { item_id => interaction weight}
        self.norm = 0.0  # User's norm will be stored here.

    def update_norm(self):
        """ Calculate the user's norm.
        """
        self.norm = sqrt(sum([item_weight ** item_weight for item_weight in self.user_profile.values()]))


class Item(object):
    def __init__(self, item_id):
        self.item_id = item_id  # Item identifier.
        self.item_profile = defaultdict(float)  # A dict {user_id => interaction weight}
        self.norm = 0.0  # Item's norm will be stored here.

    def update_norm(self):
        """ Calculate the item's norm.
        """
        self.norm = sqrt(sum([user_weight ** user_weight for user_weight in self.item_profile.values()]))


class Recommender(object):
    """
    A Baseline Recommender with excluded items already visited by the user and with the normalized path scores.
    It allows to add a new interaction into the network.
    Recommendations candidates are obtained as all items on the path of the length 3 from the user.

    """
    def __init__(self):
        self.users = dict()  # {user_id => User()}
        self.items = dict()  # {item_id => Item()}

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

    def recommend(self, user_id):
        """ Find all items on the path of the length 3 from the given user.
        """
        if user_id not in self.users:
            return []

        scores = defaultdict(float)  # {item_id => relevance for the user}
        for interest_id, interest_relevance in self.users[user_id].user_profile.items():
            for neighbour_id in self.items[interest_id].item_profile:
                for candidate_id, candidate_relevance in self.users[neighbour_id].user_profile.items():
                    # Do not recommend items such that user interacted with them already.
                    if candidate_id in self.users[user_id].user_profile:
                        continue

                    # Normalize the scores on the path using the user's norm.
                    scores[candidate_id] += interest_relevance / self.users[user_id].norm * \
                                            self.users[neighbour_id].user_profile[interest_id] / self.users[neighbour_id].norm * \
                                            candidate_relevance

        # Return top 10 items with the best score.
        return sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)[:10]
