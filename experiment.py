import json
import logging

from recommender.utils import split_dataset, evaluate, check_dataset
# from recommender.baseline_recommender import Recommender
# from recommender.recommender_exluded_history import Recommender
from recommender.normalized_recommender import Recommender
# from recommender.user_neighbours_recommender import Recommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')

dataset = json.load(open('data/dataset.json', 'r'))
logging.info('Dataset size = %d', len(dataset))

train_dataset, test_dataset = split_dataset(dataset)
logging.info('Train set size = %d', len(train_dataset))
logging.info('Test set size = %d', len(test_dataset))

check_dataset(train_dataset, test_dataset)

recommender = Recommender()
for n_interaction, interaction in enumerate(train_dataset):
    recommender.put_interaction(**interaction, weight=1.0)
logging.info('Inserted %d interactions', n_interaction + 1)

# recommender.detect_user_neighbours()

performance = evaluate(recommender, test_dataset)
logging.info('Recommender performance = %.2f%%', performance * 100)