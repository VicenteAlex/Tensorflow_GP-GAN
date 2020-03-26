import tensorflow as tf

class DataFeeder:

    def __init__(self, tfrecords_path, dataset_flag='train'):
        self._tfrecords_path = tfrecords_path
        self._dataset_flag = dataset_flag

        self._feature_dict = {
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string)
        }

    @staticmethod
    def _extract_features_batch(serialized_batch):
        """

        :param serialized_batch:
        :return:
        """
        features = tf.parse_example(
            serialized_batch,
            features={'image/height': tf.FixedLenFeature([], tf.int64),
                      'image/width': tf.FixedLenFeature([], tf.int64),
                      'image/encoded_cp': tf.FixedLenFeature([], tf.string),
                      'image/encoded_bg': tf.FixedLenFeature([], tf.string),
                      'image/format': tf.FixedLenFeature([],tf.string)
                      }
        )
        bs = features['image/encoded_cp'].shape[0]
        images_cp = tf.decode_raw(features['image/encoded_cp'], tf.uint8)
        w = features['image/width'][0]
        h = features['image/height'][0]
        images_cp = tf.cast(x=images_cp, dtype=tf.float32)

        images_cp = tf.reshape(images_cp, [bs, h, w, 3])

        images_bg = tf.decode_raw(features['image/encoded_bg'], tf.uint8)
        images_bg = tf.cast(x=images_bg, dtype=tf.float32)
        images_bg = tf.reshape(images_bg, [bs, h, w, 3])

        return images_cp, images_bg

    def inputs(self, batch_size, preprocess=None, normalize=None, name=''):
        """

        :param tfrecords_path:
        :param batch_size:
        :param num_threads:
        :return: input_images, input_labels, input_image_names
        """

        with tf.name_scope(name='inputs_' + name):

            dataset = tf.data.TFRecordDataset(self._tfrecords_path)

            dataset = dataset.shuffle(buffer_size=80000)

            # The map transformation takes a function and applies it to every element
            # of the dataset.

            dataset = dataset.batch(batch_size, drop_remainder=True)

            dataset = dataset.map(map_func=self._extract_features_batch)

            if preprocess is not None:
                dataset = dataset.map(map_func=preprocess)

            if normalize is not None:
                dataset = dataset.map(map_func=normalize)

            # The shuffle transformation uses a finite-sized buffer to shuffle elements
            # in memory. The parameter is the number of elements in the buffer. For
            # completely uniform shuffling, set the parameter to be the same as the
            # number of elements in the dataset.
            if self._dataset_flag != 'test':
                #  dataset = dataset.shuffle(buffer_size=50000)
                # repeat num epochs
                dataset = dataset.repeat()

            iterator = dataset.make_one_shot_iterator()

            out = iterator.get_next(name='{:s}_IteratorGetNext_{:s}'.format(self._dataset_flag, name))

            # map from 0 to 255 to -1 to 1
            out = list(out)
            out[0] = (out[0] / 127.5) - 1
            out[1] = (out[1] / 127.5) - 1
            out = tuple(out)

            return out
