import  numpy as np
import pandas as pd
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

def extract_images(filename):

    dataset = pd.read_csv(filename)
    num_images = len(dataset)
    print(num_images)
    rows = 8
    cols = 4
    dataset = dataset.iloc[:,0:32].values
    data = dataset.reshape(num_images, rows, cols, 1)
    return data

def extract_labels(filename):

    dataset = pd.read_csv(filename)
    num_images = len(dataset)
    datasets = dataset.astype(np.int32).values
    return datasets


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 32
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
                ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_dataset(train_dir,dtype=dtypes.float32):
    trainDataUrl="train_data.csv"
    trainLabelUrl="train_label.csv"
    testDataUrl = "test_data.csv"
    testLabelUrl = "test_label.csv"
    VALIDATION_SIZE = 5000
    train_images = extract_images(train_dir + "/" + trainDataUrl)
    train_labels = extract_labels(train_dir + "/" + trainLabelUrl)
    test_images = extract_images(train_dir + "/" + testDataUrl)
    test_labels = extract_labels(train_dir + "/" + testLabelUrl)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_images[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    train = DataSet(train_images, train_labels, dtype=dtype)
    validation = DataSet(validation_images, validation_labels, dtype=dtype)
    test = DataSet(test_images, test_labels, dtype=dtype)

    return base.Datasets(train=train, validation=validation, test=test)





