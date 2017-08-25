import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
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

    graph = sess.graph

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
#    print('Layer 3 Shape: {}'.format(vgg_layer3_out.get_shape()))
#    print('Layer 4 Shape: {}'.format(vgg_layer4_out.get_shape()))
#    print('Layer 7 Shape: {}'.format(vgg_layer7_out.get_shape()))
    decode_layer1_7T = tf.layers.conv2d_transpose(vgg_layer7_out, 512, (2, 2), (2, 2))
    # Why is a 2d convolution done on this layer? (<< Is this necessary?)
    #decode_layer1_4C = tf.layers.conv2d(vgg_layer4_out, 512, (1, 1), (1, 1))
    #decode_layer1_output = tf.add(decode_layer1_7T, decode_layer1_4C)
    # Uncomment line above or line below
    decode_layer1_output = tf.add(decode_layer1_7T, vgg_layer4_out)
    decode_layer2_L1T = tf.layers.conv2d_transpose(decode_layer1_output, 256, (2, 2), (2, 2))
    # Why is a 2d convolution done on this layer? (<< Is this necessary?)
    #decode_layer2_3C = tf.layers.conv2d(vgg_layer3_out, 256, (1, 1), (1, 1))
    #decode_layer2_output = tf.add(decode_layer2_L1T, decode_layer2_3C)
    # Uncomment line above or line below
    decode_layer2_output = tf.add(decode_layer2_L1T, vgg_layer3_out)
    decode_layer3_output = tf.layers.conv2d_transpose(decode_layer2_output, 128, (2, 2), (2, 2))
    decode_layer4_output = tf.layers.conv2d_transpose(decode_layer3_output, 64, (2, 2), (2, 2))
    decode_layer5_output = tf.layers.conv2d_transpose(decode_layer4_output, num_classes, (2, 2), (2, 2))

    return decode_layer5_output
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
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    # TODO: Implement function
    print("Training...")
    print()
    for i in range(epochs):
        batches = get_batches_fn(batch_size)
        epoch_loss = 0
        epoch_size = 0
        for batch_input, batch_label in batches:
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: batch_input,
                                                                          correct_label: batch_label,
                                                                          keep_prob: 0.5,
                                                                          learning_rate: 1e-4})
            epoch_loss += loss * len(batch_input)
            epoch_size += len(batch_input)
        print("Loss at epoch {}: {}".format(i, epoch_loss/epoch_size))
    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

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

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
