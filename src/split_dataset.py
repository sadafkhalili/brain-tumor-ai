import os
import random
import shutil

random.seed(42)

SOURCE_DIR = "data/raw"

TRAIN_DIR = "data/train"
VALID_DIR = "data/valid"
TEST_DIR = "data/test"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15


for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)


classes = os.listdir(SOURCE_DIR)

for class_name in classes:

    class_path = os.path.join(
        SOURCE_DIR,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    valid_count = int(total * VALID_RATIO)

    train_images = images[:train_count]

    valid_images = images[
        train_count:
        train_count + valid_count
    ]

    test_images = images[
        train_count + valid_count:
    ]

    for split_name, split_images in [

        ("train", train_images),

        ("valid", valid_images),

        ("test", test_images)

    ]:

        split_dir = os.path.join(
            "data",
            split_name,
            class_name
        )

        os.makedirs(
            split_dir,
            exist_ok=True
        )

        for image in split_images:

            src = os.path.join(
                class_path,
                image
            )

            dst = os.path.join(
                split_dir,
                image
            )

            shutil.copy2(src, dst)

print("Dataset Split Finished")