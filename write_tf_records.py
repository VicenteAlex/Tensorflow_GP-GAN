
import os
import tensorflow as tf
import cv2
import numpy as np
import argparse


def _int64_feature(value):
    """
    Wrapper for inserting int64 features into Example proto.
    :param value:
    :return:
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val, int):
            is_int = False
            value_tmp.append(int(float(val)))
    if not is_int:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """
    Wrapper for inserting float features into Example proto.
    :param value:
    :return:
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_float = True
    for val in value:
        if not isinstance(val, int):
            is_float = False
            value_tmp.append(float(val))
    if is_float is False:
        value = value_tmp
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """
    Wrapper for inserting bytes features into Example proto.
    :param value:
    :return:
    """
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def create_tf_example(encoded_image_data_cp, encoded_image_data_bg):

    image_format = b'jpg'

    height, width, _ = encoded_image_data_cp.shape
    encoded_image_data_cp = encoded_image_data_cp.tostring()
    encoded_image_data_bg = encoded_image_data_bg.tostring()

    feature_dict = {
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/encoded_cp': _bytes_feature(encoded_image_data_cp),
        'image/encoded_bg': _bytes_feature(encoded_image_data_bg),
        'image/format': _bytes_feature(image_format)
    }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return tf_example


def _crop(im, rw, rh, sx, sy, crop_size):

    im = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_AREA)
    im = im[sy:sy + crop_size, sx:sx + crop_size, :]
    #im = np.transpose(im, (2, 0, 1)).astype(np.float32)

    return im


def create_copy_pastes(root, folders, max_images, load_size, crop_size, crop_size_ratio, writer):

    size = int(crop_size * crop_size_ratio)
    sx_cp = crop_size // 2 - size // 2

    imgs_per_folder = {}
    for f in folders:
        folder_images = os.listdir(os.path.join(root, f))
        imgs_per_folder[f] = [os.path.join(root, f, i) for i in folder_images]

    for ii in range(max_images):
        folder = np.random.choice(folders)
        obj_path, bg_path = np.random.choice(imgs_per_folder[folder], 2, replace=False)

        obj = cv2.cvtColor(cv2.imread(obj_path), cv2.COLOR_BGR2RGB)  # source
        bg = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)  # background
        w, h, _ = obj.shape
        min_size = min(w, h)
        ratio = load_size / min_size
        rw, rh = int(np.ceil(w * ratio)), int(np.ceil(h * ratio))
        sx, sy = np.random.random_integers(0, rw - crop_size), np.random.random_integers(0, rh - crop_size)

        obj_croped = _crop(obj, rw, rh, sx, sy, crop_size)
        bg_croped = _crop(bg, rw, rh, sx, sy, crop_size)

        copy_paste = bg_croped.copy()
        copy_paste[sx_cp:sx_cp + size, sx_cp:sx_cp + size, :] = obj_croped[ sx_cp:sx_cp + size,
                                                                            sx_cp:sx_cp + size, :]

        tf_example = create_tf_example(copy_paste, bg_croped)
        writer.write(tf_example.SerializeToString())


def main():

    parser = argparse.ArgumentParser(description='Write TFRecords')
    parser.add_argument('--dataset_dir', default='DataBase/TransientAttributes/cropped_images', help='Path to cropped images')
    parser.add_argument('--out_path_train', default='DataBase/TransientAttributes/train.tfrecords', help='Output tfrecords')
    parser.add_argument('--out_path_val', default='DataBase/TransientAttributes/val.tfrecords',
                        help='Output tfrecords')
    args = parser.parse_args()

    load_size = 64
    ratio = 0.5
    image_size = 64
    val_ratio = 0.05

    writer_train = tf.python_io.TFRecordWriter(args.out_path_train)
    writer_val = tf.python_io.TFRecordWriter(args.out_path_val)

    folders = sorted(
        [folder for folder in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, folder))])
    val_end = int(val_ratio * len(folders))

    create_copy_pastes(args.dataset_dir, folders[val_end:], 150000, load_size, image_size, ratio, writer_train)
    writer_train.close()

    create_copy_pastes(args.dataset_dir, folders[:val_end], 2048, load_size, image_size, ratio, writer_val)
    writer_val.close()


if __name__ == '__main__':
    main()
