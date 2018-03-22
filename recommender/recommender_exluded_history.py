"""The same as the baseline, but items already rated by users are excluded from recommendations."""

from collections import defaultdict


class User(object):
    """ Data class that represents one user in the recommender network.
    """
    def __init__(self, user_id):
        self.user_id = user_id  # String identifier of the user.
        self.user_profile = defaultdict(float)  # A dict with users interests (interactions), { item_id => interaction weight}


class Item(object):
    """ Data class that represents one item in the recommender network.
    """
    def __init__(self, item_id):
        self.item_id = item_id  # String with the item identification.
        self.item_profile = defaultdict(float)  # {user_id => interaction weight}


class Recommender(object):
    """
    A Baseline Recommender with excluded items already visited by the user.
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

        # One item could be accessible by more then one path in the network.
        # If so, we will sum all these paths into the item's score.
        scores = defaultdict(float)
        user_node = self.users[user_id]
        for user_item_id, user_item_weight in user_node.user_profile.items():
            user_item_node = self.items[user_item_id]
            for neighbour_id, neighbour_weight in user_item_node.item_profile.items():
                neighbour_node = self.users[neighbour_id]
                for neighbour_item_id, neighbour_item_weight in neighbour_node.user_profile.items():
                    # The only one change in comparision with the baseline recommender.
                    # We will remove items already visited by the user from the recommendations.
                    if neighbour_item_id in user_node.user_profile:
                        continue

                    # Add the path weight to the item's score.
                    scores[neighbour_item_id] += user_item_weight * neighbour_weight * neighbour_item_weight

        # Return top 10 items with the best score.
        return sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)[:10]
