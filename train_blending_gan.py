from __future__ import print_function

import argparse
import os

import tensorflow as tf
import time

from model import EncoderDecoder, DCGAN_D, discriminator_loss, l2_generator_loss, generator_loss
from data_pipeline import DataFeeder

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU computation


def encode_step_cycle(step, cycle):
    return int(str(step)+str(cycle)+str(len(str(cycle))))


def decode_step_cycle(encoded):
    #  Works for numbers of cycles < 1e10
    encoded = str(encoded)
    encoded_len = int(encoded[-1])
    cycle = encoded[-(encoded_len+1):-1]
    step = encoded[:-(encoded_len+1)]
    return int(step), int(cycle)


def main():
    parser = argparse.ArgumentParser(description='Train Blending GAN')
    parser.add_argument('--nef', type=int, default=64, help='number of base filters in encoder')
    parser.add_argument('--ngf', type=int, default=64, help='number of base filters in decoder')
    parser.add_argument('--nc', type=int, default=3, help='number of output channels in decoder')
    parser.add_argument('--nBottleneck', type=int, default=4000, help='number of output channels in encoder')
    parser.add_argument('--ndf', type=int, default=64, help='number of base filters in D')

    parser.add_argument('--lr_d', type=float, default=0.0002, help='Learning rate for Critic, default=0.0002')
    parser.add_argument('--lr_g', type=float, default=0.002, help='Learning rate for Generator, default=0.002')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta for Adam, default=0.5')
    parser.add_argument('--l2_weight', type=float, default=0.99, help='Weight for l2 loss, default=0.999')
    parser.add_argument('--train_steps', default=float("58000"), help='Max amount of training cycles')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')

    parser.add_argument('--data_root',
                        default='DataBase/TransientAttributes/cropped_images',
                        help='Path to dataset')
    parser.add_argument('--train_data_root', default='DataBase/TransientAttributes/train.tfrecords', help='Path to train dataset')
    parser.add_argument('--val_data_root', default='DataBase/TransientAttributes/val.tfrecords', help='Path to val dataset')
    parser.add_argument('--image_size', type=int, default=64, help='The height / width of the network\'s input image')

    parser.add_argument('--d_iters', type=int, default=5, help='# of discriminator iters per each generator iter')
    parser.add_argument('--clamp_lower', type=float, default=-0.01, help='Lower bound for weight clipping')
    parser.add_argument('--clamp_upper', type=float, default=0.01, help='Upper bound for weight clipping')

    parser.add_argument('--experiment', default='blending_gan',
                        help='Where to store samples and models')
    parser.add_argument('--save_folder', default='GP-GAN_training', help='location to save')
    parser.add_argument('--tboard_save_dir', default='tensorboard', help='location to save tboard records')

    parser.add_argument('--val_freq', type=int, default=500, help='frequency of validation')
    parser.add_argument('--snapshot_interval', type=int, default=500, help='Interval of snapshot (steps)')

    parser.add_argument('--weights_path', type=str, default=
                        None, help='path to checkpoint')

    args = parser.parse_args()

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    # Set up generator & discriminator
    print('Create & Init models ...')
    print('\tInit Generator network ...')
    generator = EncoderDecoder(encoder_filters=args.nef, encoded_dims=args.nBottleneck, output_channels=args.nc,
                               decoder_filters=args.ngf, is_training=True, image_size=args.image_size, skip=False, scope_name='generator') #, conv_init=init_conv,

    generator_val = EncoderDecoder(encoder_filters=args.nef, encoded_dims=args.nBottleneck, output_channels=args.nc,
                               decoder_filters=args.ngf, is_training=False, image_size=args.image_size, skip=False, scope_name='generator')

    print('\tInit Discriminator network ...')
    discriminator = DCGAN_D(image_size=args.image_size, encoded_dims=1, filters=args.ndf, is_training=True, scope_name='discriminator') #, conv_init=init_conv, bn_init=init_bn)  # D

    discriminator_val = DCGAN_D(image_size=args.image_size, encoded_dims=1, filters=args.ndf, is_training=False,
                           scope_name='discriminator')

    # Set up training graph
    with tf.device('/gpu:0'):

        train_dataset = DataFeeder(tfrecords_path=args.train_data_root, dataset_flag='train')
        composed_image, real_image = train_dataset.inputs(batch_size=args.batch_size, name='train_dataset')
        shape = composed_image.get_shape().as_list()
        composed_image.set_shape([shape[0], args.image_size, args.image_size, shape[3]])
        real_image.set_shape([shape[0], args.image_size, args.image_size, shape[3]])

        validation_dataset = DataFeeder(tfrecords_path=args.val_data_root, dataset_flag='val')
        composed_image_val, real_image_val = validation_dataset.inputs(batch_size=args.batch_size, name='val_dataset')
        composed_image_val.set_shape([shape[0], args.image_size, args.image_size, shape[3]])
        real_image_val.set_shape([shape[0], args.image_size, args.image_size, shape[3]])

        # Compute losses:

        # Train tensors
        fake = generator(composed_image)
        prob_disc_real = discriminator.encode(real_image)
        prob_disc_fake = discriminator.encode(fake)

        # Validation tensors
        fake_val = generator_val(composed_image)
        prob_disc_real_val = discriminator_val.encode(real_image)
        prob_disc_fake_val = discriminator_val.encode(fake)

        # Calculate losses
        gen_loss, l2_comp, disc_comp, fake_image_train = l2_generator_loss(fake=fake, target=real_image, prob_disc_fake=prob_disc_fake, l2_weight=args.l2_weight)

        disc_loss = discriminator_loss(prob_disc_real=prob_disc_real, prob_disc_fake=prob_disc_fake)

        gen_loss_val, _, _, fake_image_val = l2_generator_loss(fake=fake_val, target=real_image, prob_disc_fake=prob_disc_fake_val, l2_weight=args.l2_weight)

        disc_loss_val = discriminator_loss(prob_disc_real=prob_disc_real_val, prob_disc_fake=prob_disc_fake_val)

        # Set optimizers
        global_step = tf.Variable(0, name='global_step', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):

            discriminator_variables = [v for v in tf.trainable_variables() if v.name.startswith("discriminator")]
            generator_variables = [v for v in tf.trainable_variables() if v.name.startswith("generator")]

            optimizer_gen = tf.train.AdamOptimizer(learning_rate=args.lr_g, beta1=args.beta1).minimize(
                loss=gen_loss, global_step=global_step, var_list=generator_variables)

            optimizer_disc = tf.train.AdamOptimizer(learning_rate=args.lr_d, beta1=args.beta1).minimize(
                loss=disc_loss, global_step=global_step, var_list=discriminator_variables)


            with tf.name_scope("clip_weights"):
                clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, args.clamp_lower, args.clamp_upper)) for
                                             var in discriminator_variables]

    # Set summaries for Tensorboard

    model_save_dir = os.path.join(args.save_folder, args.experiment)

    tboard_save_dir = os.path.join(model_save_dir, args.tboard_save_dir)
    os.makedirs(tboard_save_dir, exist_ok=True)
    sum_gen_train = tf.summary.scalar(name='train_gen_loss', tensor=gen_loss)
    sum_gen_disc_comp = tf.summary.scalar(name='train_gen_disc_component', tensor=disc_comp)
    sum_gen_l2_comp = tf.summary.scalar(name='train_gen_l2_component', tensor=l2_comp)

    sum_gen_val = tf.summary.scalar(name='val_gen_loss', tensor=gen_loss_val, collections='')
    sum_disc_train = tf.summary.scalar(name='train_disc_loss', tensor=disc_loss)
    sum_disc_val = tf.summary.scalar(name='val_disc_loss', tensor=disc_loss_val)
    sum_fake_image_train = tf.summary.image(name='train_image_generated', tensor=fake_image_train)
    sum_fake_image_val = tf.summary.image(name='val_image_generated', tensor=fake_image_val)
    sum_disc_real = tf.summary.scalar(name='train_disc_value_real', tensor=tf.reduce_mean(prob_disc_real))
    sum_disc_fake = tf.summary.scalar(name='train_disc_value_fake', tensor=tf.reduce_mean(prob_disc_fake))

    sum_composed = tf.summary.image(name='composed', tensor=composed_image)
    sum_real = tf.summary.image(name='real', tensor=real_image)
    
    train_merge = tf.summary.merge([sum_gen_train, sum_fake_image_train, sum_disc_train, sum_composed, sum_real,
                                    sum_gen_disc_comp, sum_gen_l2_comp, sum_disc_real, sum_disc_fake])

    # Set saver configuration

    loader = tf.train.Saver()
    saver = tf.train.Saver()
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'GP-GAN_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(model_save_dir, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=sess_config)

    # Write graph to tensorboard
    summary_writer = tf.summary.FileWriter(tboard_save_dir)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters

    with sess.as_default():
        step = 0
        cycle = 0

        if args.weights_path is None:
            print('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            print('Restore model from {:s}'.format(args.weights_path))
            loader.restore(sess=sess, save_path=args.weights_path)

            step_cycle = args.weights_path.split('ckpt-')[-1]
            step, cycle = decode_step_cycle(step_cycle)

        gen_train_loss = '?'
        while cycle <= args.train_steps:

            # (1) Update discriminator network
            # train the discriminator Diters times

            if cycle < 25 or cycle % 500 == 0:
                Diters = 100

            else:
                Diters = args.d_iters

            for _ in range(Diters):
                # enforce Lipschitz constraint
                sess.run(clip_discriminator_var_op)

                _, disc_train_loss = sess.run([optimizer_disc, disc_loss])
                print('Step: ' + str(step) + ' Cycle: ' + str(cycle) + ' Train discriminator loss: '
                      + str(disc_train_loss) + ' Train generator loss: ' + str(gen_train_loss))

                step += 1

            # (2) Update generator network

            _, gen_train_loss, train_merge_value = sess.run([optimizer_gen, gen_loss, train_merge])
            summary_writer.add_summary(summary=train_merge_value, global_step=cycle)

            if cycle != 0 and cycle % args.val_freq == 0:
                _, disc_val_loss, gen_val_value, fake_image_val_value = sess.run([optimizer_disc, gen_loss_val, sum_gen_val, sum_fake_image_val])
                _, gen_val_loss, disc_val_value = sess.run([optimizer_gen, disc_loss_val, sum_disc_val])
                print('Step: ' + str(step) + ' Cycle: ' + str(cycle) + ' Val discriminator loss: ' + str(disc_val_loss)
                      + ' Val generator loss: ' + str(gen_val_loss))
                summary_writer.add_summary(summary=gen_val_value, global_step=cycle)
                summary_writer.add_summary(summary=disc_val_value, global_step=cycle)
                summary_writer.add_summary(summary=fake_image_val_value, global_step=cycle)

            if cycle != 0 and cycle % args.snapshot_interval == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=encode_step_cycle(step, cycle))
            cycle += 1


if __name__ == '__main__':
    main()
