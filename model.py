import tensorflow as tf

bn_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
conv_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)


def discriminator_loss(prob_disc_real, prob_disc_fake):

    with tf.name_scope("discriminator_loss"):
        return tf.reduce_mean(prob_disc_real) - tf.reduce_mean(prob_disc_fake)  # maps real images to negative and fake to positive to minimize cost


def generator_loss(prob_disc_fake):
    with tf.name_scope("generator_loss"):
        return tf.reduce_mean(prob_disc_fake)


def l2_generator_loss(fake, target, prob_disc_fake, l2_weight):

    with tf.name_scope("l2_generator_loss"):
        l2_comp = (l2_weight*tf.losses.mean_squared_error(target, fake))
        disc_comp = ((1-l2_weight)*tf.reduce_mean(prob_disc_fake))
        return l2_comp + disc_comp, l2_comp, disc_comp, fake


def conv2d(inputdata, out_channel, kernel_size, padding='SAME',
           stride=1, w_init=None, b_init=None,
           split=1, use_bias=True, data_format='NHWC', name=None):
    """
    Packing the tensorflow conv2d function.
    :param name: op name
    :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
    unknown dimensions.
    :param out_channel: number of output channel.
    :param kernel_size: int so only support square kernel convolution
    :param padding: 'VALID' or 'SAME'
    :param stride: int so only support square stride
    :param w_init: initializer for convolution weights
    :param b_init: initializer for bias
    :param split: split channels as used in Alexnet mainly group for GPU memory save.
    :param use_bias:  whether to use bias.
    :param data_format: default set to NHWC according tensorflow
    :return: tf.Tensor named ``output``
    """
    with tf.variable_scope(name):
        in_shape = inputdata.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]

        assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
        assert in_channel % split == 0
        assert out_channel % split == 0

        padding = padding.upper()

        if isinstance(kernel_size, list):
            filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel / split, out_channel]
        else:
            filter_shape = [kernel_size, kernel_size] + [in_channel / split, out_channel]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                else [1, 1, stride, stride]

        if w_init is None:
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        w = tf.get_variable('W', filter_shape, initializer=w_init)
        b = None

        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=b_init)

        if split == 1:
            conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format=data_format)
        else:
            inputs = tf.split(inputdata, split, channel_axis)
            kernels = tf.split(w, split, 3)
            outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                       for i, k in zip(inputs, kernels)]
            conv = tf.concat(outputs, channel_axis)

        ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format)
                          if use_bias else conv, name=name)

    return ret


def deconv2d(inputdata, out_channel, kernel_size, padding='SAME',
             stride=1, w_init=None, b_init=None,
             use_bias=True, activation=None, data_format='channels_last',
             trainable=True, name=None):
    """
    Packing the tensorflow conv2d function.
    :param name: op name
    :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
    unknown dimensions.
    :param out_channel: number of output channel.
    :param kernel_size: int so only support square kernel convolution
    :param padding: 'VALID' or 'SAME'
    :param stride: int so only support square stride
    :param w_init: initializer for convolution weights
    :param b_init: initializer for bias
    :param activation: whether to apply a activation func to deconv result
    :param use_bias:  whether to use bias.
    :param data_format: default set to NHWC according tensorflow
    :param trainable:
    :return: tf.Tensor named ``output``
    """
    with tf.variable_scope(name):
        in_shape = inputdata.get_shape().as_list()
        channel_axis = 3 if data_format == 'channels_last' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"

        padding = padding.upper()

        if w_init is None:
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        ret = tf.layers.conv2d_transpose(inputs=inputdata, filters=out_channel,
                                         kernel_size=kernel_size,
                                         strides=stride, padding=padding,
                                         data_format=data_format,
                                         activation=activation, use_bias=use_bias,
                                         kernel_initializer=w_init,
                                         bias_initializer=b_init, trainable=trainable,
                                         name=name)
    return ret


class DCGAN_G():  # decoder
    def __init__(self, image_size, output_channels, base_filters, is_training, scope_name=None):
        self.image_size = image_size
        self.base_filters = base_filters # ngf
        self.output_channels = output_channels # nc
        self.scope_name = scope_name
        self.is_training = is_training

    def decode(self, x):
        with tf.variable_scope(name_or_scope=self.scope_name, reuse= tf.AUTO_REUSE):
            cngf, tisize = self.base_filters // 2, 4
            while tisize != self.image_size:
                cngf = cngf * 2
                tisize = tisize * 2

            x = deconv2d(x, out_channel=cngf, kernel_size=4, stride=1, padding='VALID', name='dconv_1', w_init=conv_init, use_bias=False)
            x = tf.layers.batch_normalization(inputs=x, training=self.is_training, name='bn_1', momentum=0.99, epsilon=2e-5, gamma_initializer=bn_init)
            x = tf.nn.relu(features=x, name='relu_1')

            csize, cndf = 4, cngf
            i=2
            while csize < self.image_size // 2:
                x = deconv2d(x, out_channel=cngf // 2, kernel_size=4, stride=2, padding='SAME', name='dconv_'+str(i), w_init=conv_init, use_bias=False)
                x = tf.layers.batch_normalization(inputs=x, training=self.is_training, name='bn_'+str(i), momentum=0.99, epsilon=2e-5, gamma_initializer=bn_init)
                x = tf.nn.relu(features=x, name='relu_'+str(i))

                cngf = cngf // 2
                csize = csize * 2
                i += 1
            x = deconv2d(x, out_channel=self.output_channels, kernel_size=4, stride=2, padding='SAME', use_bias=False, name='dconv_'+str(i), w_init=conv_init)
            x = tf.tanh(x, name='final_tanh')
        return x


class DCGAN_G_skip():  # decoder
    def __init__(self, image_size, output_channels, base_filters, is_training, scope_name=None):
        self.image_size = image_size
        self.base_filters = base_filters # ngf
        self.output_channels = output_channels # nc
        self.scope_name = scope_name
        self.is_training = is_training

    def decode(self, x):
        with tf.variable_scope(name_or_scope=self.scope_name, reuse= tf.AUTO_REUSE):
            skips = x[1]
            x = x[0]
            cngf, tisize = self.base_filters // 2, 4
            while tisize != self.image_size:
                cngf = cngf * 2
                tisize = tisize * 2

            x = deconv2d(x, out_channel=cngf, kernel_size=4, stride=1, padding='VALID', name='dconv_1', w_init=conv_init, use_bias=False)
            x = tf.layers.batch_normalization(inputs=x, training=self.is_training, name='bn_1', momentum=0.99, epsilon=2e-5, gamma_initializer=bn_init)
            x = tf.nn.relu(features=x, name='relu_1')

            x = x + skips[-1]

            csize, cndf = 4, cngf
            i=2
            while csize < self.image_size // 2:
                x = deconv2d(x, out_channel=cngf // 2, kernel_size=4, stride=2, padding='SAME', name='dconv_'+str(i), w_init=conv_init, use_bias=False)
                x = tf.layers.batch_normalization(inputs=x, training=self.is_training, name='bn_'+str(i), momentum=0.99, epsilon=2e-5, gamma_initializer=bn_init)
                x = tf.nn.relu(features=x, name='relu_'+str(i))
                x = x + skips[-i]

                cngf = cngf // 2
                csize = csize * 2
                i += 1
            x = deconv2d(x, out_channel=self.output_channels, kernel_size=4, stride=2, padding='SAME', use_bias=False, name='dconv_'+str(i), w_init=conv_init)
            x = tf.tanh(x, name='final_tanh')
        return x


class DCGAN_D():  # encoder
    def __init__(self, filters, image_size, encoded_dims, is_training, scope_name=None):
        self.filters = filters
        self.image_size = image_size
        self.encoded_dims = encoded_dims
        self.scope_name = scope_name
        self.is_training = is_training

    def encode(self, x):
        with tf.variable_scope(name_or_scope=self.scope_name, reuse= tf.AUTO_REUSE):
            x = conv2d(x, self.filters, 4, padding='SAME', stride=2, name='conv_1', w_init=conv_init, use_bias=False)
            x = tf.nn.leaky_relu(x, name='leaky_relu_1')

            csize, cndf = self.image_size / 2, self.filters
            i=2
            while csize > 4:
                out_feat = cndf * 2
                x = conv2d(x, out_feat, 4, padding='SAME', stride=2, name='conv_'+str(i), w_init=conv_init, use_bias=False)
                x = tf.layers.batch_normalization(inputs=x, training=self.is_training, name='bn_'+str(i), momentum=0.99, epsilon=2e-5,gamma_initializer=bn_init)
                x = tf.nn.leaky_relu(x, name='leaky_relu_'+str(i))

                cndf = cndf * 2
                csize = csize / 2
                i+=1
            # state size. K x 4 x 4
            x = conv2d(x, self.encoded_dims, 4, padding='VALID', stride=1, use_bias=False, name='conv_'+str(i), w_init=conv_init)  # Does the same than flatten + FC of 1 output

        return x


class DCGAN_D_skip():  # encoder
    def __init__(self, filters, image_size, encoded_dims, is_training, scope_name=None):
        self.filters = filters
        self.image_size = image_size
        self.encoded_dims = encoded_dims
        self.scope_name = scope_name
        self.is_training = is_training

    def encode(self, x):
        with tf.variable_scope(name_or_scope=self.scope_name, reuse= tf.AUTO_REUSE):

            skips = []

            x = conv2d(x, self.filters, 4, padding='SAME', stride=2, name='conv_1', w_init=conv_init, use_bias=False)
            x = tf.nn.leaky_relu(x, name='leaky_relu_1')
            skips.append(x)

            csize, cndf = self.image_size / 2, self.filters
            i=2
            while csize > 4:
                out_feat = cndf * 2
                x = conv2d(x, out_feat, 4, padding='SAME', stride=2, name='conv_'+str(i), w_init=conv_init, use_bias=False)
                x = tf.layers.batch_normalization(inputs=x, training=self.is_training, name='bn_'+str(i), momentum=0.99, epsilon=2e-5,gamma_initializer=bn_init)
                x = tf.nn.leaky_relu(x, name='leaky_relu_'+str(i))
                skips.append(x)

                cndf = cndf * 2
                csize = csize / 2
                i+=1
            # state size. K x 4 x 4
            x = conv2d(x, self.encoded_dims, 4, padding='VALID', stride=1, use_bias=False, name='conv_'+str(i), w_init=conv_init)  # Does the same than flatten + FC of 1 output

        return x, skips


class EncoderDecoder():
    def __init__(self, encoder_filters, encoded_dims, output_channels, decoder_filters, is_training, image_size=64, skip=False, scope_name=None):
        self.skip = skip
        if skip:
            self.encoder = DCGAN_D_skip(encoder_filters, image_size, encoded_dims, is_training, scope_name + '/encoder')
            self.decoder = DCGAN_G_skip(image_size, output_channels, decoder_filters, is_training, scope_name + '/decoder')
        else:
            self.encoder = DCGAN_D(encoder_filters, image_size, encoded_dims, is_training, scope_name + '/encoder')
            self.decoder = DCGAN_G(image_size, output_channels, decoder_filters, is_training, scope_name + '/decoder')
        self.is_training = is_training
        self.scope_name = scope_name

    def encode(self, x):
        if self.skip:
            h, skips = self.encoder.encode(x)
        else:
            h = self.encoder.encode(x)
        with tf.variable_scope(name_or_scope=self.scope_name, reuse= tf.AUTO_REUSE):
            h = tf.layers.batch_normalization(inputs=h, training=self.is_training, name='bn_last', momentum=0.99, epsilon=2e-5, gamma_initializer=bn_init)
            h = tf.nn.leaky_relu(h, name='leaky_relu_last')
        if self.skip:
            return h, skips
        else:
            return h

    def decode(self, x):
        h = self.decoder.decode(x)

        return h

    def __call__(self, x):
        h = self.encode(x)
        h = self.decode(h)
        return h

