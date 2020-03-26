import argparse
import os

import cv2
import tensorflow as tf

from gp_gan import gp_gan
from model import EncoderDecoder

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU computation

basename = lambda path: os.path.splitext(os.path.basename(path))[0]

"""
    Note: source image, destination image and mask image have the same size.
"""


def main():
    parser = argparse.ArgumentParser(description='Gaussian-Poisson GAN for high-resolution image blending')
    parser.add_argument('--nef', type=int, default=64, help='# of base filters in encoder')
    parser.add_argument('--ngf', type=int, default=64, help='# of base filters in decoder or G')
    parser.add_argument('--nc', type=int, default=3, help='# of output channels in decoder or G')
    parser.add_argument('--nBottleneck', type=int, default=4000, help='# of output channels in encoder')
    parser.add_argument('--ndf', type=int, default=64, help='# of base filters in D')

    parser.add_argument('--image_size', type=int, default=64, help='The height / width of the input image to network')

    parser.add_argument('--color_weight', type=float, default=1, help='Color weight')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Sigma for gaussian smooth of Gaussian-Poisson Equation')
    parser.add_argument('--gradient_kernel', type=str, default='normal', help='Kernel type for calc gradient')
    parser.add_argument('--smooth_sigma', type=float, default=1, help='Sigma for gaussian smooth of Laplacian pyramid')

    parser.add_argument('--generator_path', default=None, help='Path to GAN model checkpoint')

    parser.add_argument('--list_path', default='',
                        help='File for input list in csv format: obj_path;bg_path;mask_path in each line')
    parser.add_argument('--result_folder', default='blending_result', help='Name for folder storing results')

    parser.add_argument('--src_image', default='DataBase/test_images/src.jpg', help='Path for source image')
    parser.add_argument('--dst_image', default='DataBase/test_images/dst.jpg', help='Path for destination image')
    parser.add_argument('--mask_image', default='DataBase/test_images/mask.png', help='Path for mask image')
    parser.add_argument('--blended_image', default='DataBase/test_images/result2.jpg', help='Where to save blended image')

    args = parser.parse_args()

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    # Init CNN model
    generator = EncoderDecoder(encoder_filters=args.nef, encoded_dims=args.nBottleneck, output_channels=args.nc,
                               decoder_filters=args.ngf, is_training=False, image_size=args.image_size,
                               scope_name='generator')

    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, args.image_size, args.image_size, args.nc],
        name='input'
    )

    gan_im_tens = generator(inputdata)

    loader = tf.train.Saver(tf.all_variables())
    sess = tf.Session()

    with sess.as_default():
        loader.restore(sess=sess, save_path=args.generator_path)

    # Init image list
    if args.list_path:
        print('Load images from {} ...'.format(args.list_path))
        with open(args.list_path) as f:
            test_list = [line.strip().split(';') for line in f]
        print('\t {} images in total ...\n'.format(len(test_list)))
    else:
        test_list = [(args.src_image, args.dst_image, args.mask_image)]

    if not args.blended_image:
        # Init result folder
        if not os.path.isdir(args.result_folder):
            os.makedirs(args.result_folder)
        print('Result will save to {} ...\n'.format(args.result_folder))

    total_size = len(test_list)
    for idx in range(total_size):
        print('Processing {}/{} ...'.format(idx + 1, total_size))

        # load image
        obj = cv2.cvtColor(cv2.imread(test_list[idx][0], 1), cv2.COLOR_BGR2RGB) / 255
        bg = cv2.cvtColor(cv2.imread(test_list[idx][1], 1), cv2.COLOR_BGR2RGB) / 255
        mask = cv2.imread(test_list[idx][2], 0).astype(obj.dtype)

        blended_im = gp_gan(obj, bg, mask, gan_im_tens, inputdata, sess, args.image_size, color_weight=args.color_weight,
                            sigma=args.sigma,
                            gradient_kernel=args.gradient_kernel, smooth_sigma=args.smooth_sigma)

        if args.blended_image:
            cv2.imwrite(args.blended_image, cv2.cvtColor(blended_im, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite('{}/obj_{}_bg_{}_mask_{}.png'.format(args.result_folder, basename(test_list[idx][0]),
                                                        basename(test_list[idx][1]), basename(test_list[idx][2])),
                   blended_im)


if __name__ == '__main__':
    main()
