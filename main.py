# !/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    w2 = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w5 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, w2, w3, w4, w5


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # FCN-8 - Decoder
    # https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/595f35e6-b940-400f-afb2-2015319aa640/lessons/69fe4a9c-656e-46c8-bc32-aee9e60b8984/concepts/3dcaf318-9e4b-4bb6-b057-886c254abd44
    #

    with tf.name_scope("1x1"):
        conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes,
                                    kernel_size=1,
                                    padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                    # , name="conv_1x1"
                                    )

    with tf.name_scope("decoder_1"):
        # upsample by 2
        output = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2,
                                            padding='same',
                                            # kernel_initializer=
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                            # , name="Upsample_1"
                                            )

        # add skip layer
        vgg_layer4_out_conv = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                               padding='same',
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                               # , name="Layer_4_skip"
                                               )
        output = tf.add(vgg_layer4_out_conv, output)

    with tf.name_scope("decoder_2"):
        # upsample by 2
        output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, padding='same',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                            # , name="Upsample_2"
                                            )

        # add skip layer
        vgg_layer3_out_conv = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                               padding='same',
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                               # , name="Layer_3_skip"
                                               )

        output = tf.add(vgg_layer3_out_conv, output)

    with tf.name_scope("decoder_3"):
        # upsample by 8
        output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='same',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                            # , name="Upsample_3"
                                            )

    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # FCN-8 - Classification & Loss
    # https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/595f35e6-b940-400f-afb2-2015319aa640/lessons/69fe4a9c-656e-46c8-bc32-aee9e60b8984/concepts/c9cbe9d0-22c1-4362-bdaa-282a124ca852
    #

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    with tf.name_scope("xent"):
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
        combined_loss = cross_entropy_loss + 1e-3 * sum(regularization_loss)
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(combined_loss)
    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, logits):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # Compute the accuracy
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(tf.arg_max(logits, 1), tf.arg_max(correct_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        epoch_loss = 0
        batches = 0
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image,
                                          correct_label: label,
                                          keep_prob: 0.5,
                                          learning_rate: 0.0009})
            epoch_loss += loss
            batches += 1
        print("Loss: = {:.3f}".format(epoch_loss / batches))
        print("Accuracy: = {:.3f}".format(accuracy))


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 50
    batch_size = 25

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input, keep, layer_3, layer_4, layer_7 = load_vgg(sess, vgg_path)
        output = layers(layer_3, layer_4, layer_7, num_classes)

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input,
                 correct_label, keep, learning_rate, logits)

        # Save data for tensorboard
        output_dir = os.path.join(helper.folder_for_current_run(runs_dir), "log")
        writer = tf.summary.FileWriter(output_dir)
        writer.add_graph(sess.graph)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep, input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
