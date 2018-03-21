def load_dataset(filename):
    """ Load specified dataset from CSV, provide some data postprocessing.
    """
    dataset = []
    with open(filename, 'r') as raw_data:
        for n_line, line in enumerate(raw_data):
            if not n_line:
                continue

            data_fields = line.rstrip().split(';')

            user_id = data_fields[0][1:-1]
            item_id = data_fields[1][1:-1]
            rating = float(data_fields[2][1:-1]) / 10

            if not rating:
                continue

            dataset.append({
                'user_id': user_id,
                'item_id': item_id
            })

    return dataset


def anonymize_dataset(dataset):
    """ Change user and item identifiers to a new one.
    """
    anonymized_dataset = []

    user_offset2id = []
    item_offset2id = []
    user_id2offset = {}
    item_id2offset = {}

    for interaction in dataset:
        user_id = interaction['user_id']
        item_id = interaction['item_id']

        if user_id not in user_id2offset:
            user_id2offset[user_id] = len(user_offset2id)
            user_offset2id.append(user_id)

        if item_id not in item_id2offset:
            item_id2offset[item_id] = len(item_offset2id)
            item_offset2id.append(item_id)

        anonymized_dataset.append({
            'user_id': 'user_{:04}'.format(user_id2offset[user_id]),
            'item_id': 'item_{:04}'.format(item_id2offset[item_id])
        })

    return anonymized_dataset
