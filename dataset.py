import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def get_dataset(
        images: list,
        labels: list,
        train_size: float = 0.8,
        seed: int = 32
):
    print("Preparing dataset...")

    train_data, test_data, train_labels, test_labels = train_test_split(images,
                                                                        labels,
                                                                        train_size=train_size,
                                                                        random_state=seed)
    print(
        f"Number of training samples: {len(train_labels)} \n",
        f"Number of testing samples: {len(test_labels)} \n"
    )

    train_data = np.array(train_data) / 255.0
    test_data = np.array(test_data) / 255.0
    train_labels_text = np.array(train_labels)
    test_labels_text = np.array(test_labels)

    lb = LabelBinarizer()
    lb.fit(train_labels_text)

    train_labels = lb.transform(train_labels_text)
    test_labels = lb.transform(test_labels_text)

    divide_point = int(0.8 * len(train_data))

    trainX = train_data[:divide_point]
    trainY = train_labels[:divide_point]
    valX = train_data[divide_point:]
    valY = train_labels[divide_point:]
    testX = test_data
    testY = test_labels

    print("Dataset is ready")
    return trainX, trainY, valX, valY, testX, testY
