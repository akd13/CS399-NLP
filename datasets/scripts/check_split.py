import json
split_files = ['../downsampled_images/images.json', '../images/images.json']
for file in split_files:
    with open(file) as f:
        data = json.load(f)

    train_count = 0
    test_count = 0
    val_count = 0

    for image in data['images']:
        if image['split'] == 'train':
            train_count += 1
        elif image['split'] == 'test':
            test_count += 1
        elif image['split'] == 'val':
            val_count += 1

    print(f"Train count: {train_count}")
    print(f"Test count: {test_count}")
    print(f"Val count: {val_count}")
