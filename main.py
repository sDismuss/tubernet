# IMPORT LIBRARIES
import os

import tensorflow as tf
from imutils import paths

import config
import dataset
import model
import preprocess_images


def main(args):
    ##########################################################################
    #                             Basic settings                             #
    ##########################################################################

    TRAIN_SIZE = 0.8
    IMG_SIZE = (args.width, args.height)
    INPUT_SIZE = (args.width, args.height, 3)
    IMAGE_PATHS = list(paths.list_images(args.data_dir))
    NUM_CLASSES = len(os.listdir(args.data_dir))

    ##########################################################################
    #                          Image preprocessing                           #
    ##########################################################################

    images, labels = preprocess_images.get_preprocessed_images(
        image_paths=IMAGE_PATHS,
        image_size=IMG_SIZE
    )

    ##########################################################################
    #                           Preparing dataset                            #
    ##########################################################################
    trainX, trainY, valX, valY, testX, testY = dataset.get_dataset(
        images=images,
        labels=labels,
        train_size=TRAIN_SIZE,
        seed=args.seed
    )

    ##########################################################################
    #                             Creating model                             #
    ##########################################################################
    tubernet = model.get_model(
        num_classes=NUM_CLASSES,
        learning_rate=args.lr,
        input_size=INPUT_SIZE,
        seed=args.seed
    )
    tubernet.summary()

    ##########################################################################
    #                            Learning model                              #
    ##########################################################################
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "archive/vgg16-chestx-ray.hdf5",
        save_weights_only=False,
        save_best_only=True,
        monitor="val_accuracy"
    )

    history = tubernet.fit(
        trainX,
        trainY,
        batch_size=args.batches,
        epochs=args.epochs,
        validation_data=(valX, valY),
        callbacks=[checkpoint]
    )

    ##########################################################################
    #                          Result calculation                            #
    ##########################################################################
    best = tf.keras.models.load_model("archive/vgg16-chestx-ray.hdf5")

    val_loss, val_acc = best.evaluate(valX, valY)
    test_loss, test_acc = best.evaluate(testX, testY)

    print("Validation Accuracy: {:.2f} %".format(100 * val_acc))
    print("Validation Loss: {:.2f} %".format(100 * val_loss))
    print("\nTest Accuracy: {:.2f} %".format(100 * test_acc))
    print("Test Loss: {:.2f} %".format(100 * test_loss))


if __name__ == "__main__":
    args = config.parse_arguments()
    main(args)
