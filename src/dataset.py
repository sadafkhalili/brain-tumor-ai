import os
import random
import shutil

random.seed(42)

SOURCE_DIR = "data/raw"

TRAIN_DIR = "data/train"
VALID_DIR = "data/valid"
TEST_DIR = "data/test"

classes = os.listdir(SOURCE_DIR)

for class_name in classes:

    source_class = os.path.join(
        SOURCE_DIR,
        class_name
    )

    images = os.listdir(source_class)

    random.shuffle(images)

    train_size = int(0.70 * len(images))
    valid_size = int(0.15 * len(images))

    train_images = images[:train_size]

    valid_images = images[
        train_size:
        train_size + valid_size
    ]

    test_images = images[
        train_size + valid_size:
    ]

    os.makedirs(
        os.path.join(TRAIN_DIR, class_name),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(VALID_DIR, class_name),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(TEST_DIR, class_name),
        exist_ok=True
    )

    for image in train_images:

        shutil.copy(
            os.path.join(source_class, image),
            os.path.join(
                TRAIN_DIR,
                class_name,
                image
            )
        )

    for image in valid_images:

        shutil.copy(
            os.path.join(source_class, image),
            os.path.join(
                VALID_DIR,
                class_name,
                image
            )
        )

    for image in test_images:

        shutil.copy(
            os.path.join(source_class, image),
            os.path.join(
                TEST_DIR,
                class_name,
                image
            )
        )

print("Dataset Split Finished")