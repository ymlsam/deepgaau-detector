import tensorflow as tf

from object_detection import model_lib
from object_detection.core import standard_fields as fields
from object_detection.core.model import DetectionModel
from object_detection.utils import ops
from typing import Dict, Tuple


def predict_with_loss(
        model: DetectionModel,
        features: Dict,
        labels: Dict,
        add_regularization_loss: bool = True,
) -> Tuple[Dict, Dict]:
    """Computes the losses dict and predictions dict for a model on inputs.
    
    Args:
        model: a DetectionModel (based on Keras).
        features: Dictionary of feature tensors from the input dataset.
            Should be in the format output by `inputs.train_input` and `inputs.eval_input`.
            features[fields.InputDataFields.image] is a [batch_size, H, W, C] float32 tensor with preprocessed images.
            features[HASH_KEY] is a [batch_size] int32 tensor representing unique identifiers for the images.
            features[fields.InputDataFields.true_image_shape] is a [batch_size, 3] int32 tensor representing the true
            image shapes, as preprocessed images could be padded.
            features[fields.InputDataFields.original_image] (optional) is a [batch_size, H, W, C] float32 tensor with
            original images.
        labels: A dictionary of groundtruth tensors post-unstacking.
            The original labels are of the form returned by `inputs.train_input` and `inputs.eval_input`.
            The shapes may have been modified by unstacking with `model_lib.unstack_batch`.
            However, the dictionary includes the following fields.
            labels[fields.InputDataFields.num_groundtruth_boxes] is a int32 tensor indicating the number of valid
            groundtruth boxes per image.
            labels[fields.InputDataFields.groundtruth_boxes] is a float32 tensor containing the corners of the
            groundtruth boxes.
            labels[fields.InputDataFields.groundtruth_classes] is a float32 one-hot tensor of classes.
            labels[fields.InputDataFields.groundtruth_weights] is a float32 tensor containing groundtruth weights for
            the boxes.
            -- Optional --
            labels[fields.InputDataFields.groundtruth_instance_masks] is a float32 tensor containing only binary values,
            which represent instance masks for objects.
            labels[fields.InputDataFields.groundtruth_keypoints] is a float32 tensor containing keypoints for each box.
            labels[fields.InputDataFields.groundtruth_dp_num_points] is an int32 tensor with the number of sampled
            DensePose points per object.
            labels[fields.InputDataFields.groundtruth_dp_part_ids] is an int32 tensor with the DensePose part ids
            (0-indexed) per object.
            labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a float32 tensor with the DensePose surface
            coordinates.
            labels[fields.InputDataFields.groundtruth_group_of] is a tf.bool tensor containing group_of annotations.
            labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32 k-hot tensor of classes.
            labels[fields.InputDataFields.groundtruth_track_ids] is a int32 tensor of track IDs.
        add_regularization_loss: Whether or not to include the model's regularization loss in the losses dictionary.
    
    Returns:
        A tuple containing the losses dictionary (with the total loss under the key 'Loss/total_loss'), and the
        predictions dictionary produced by `model.predict`.
    """
    
    # predict
    model_lib.provide_groundtruth(model, labels)
    preprocessed_images = features[fields.InputDataFields.image]
    true_image_shape = features[fields.InputDataFields.true_image_shape]
    side_inputs = model.get_side_inputs(features)
    
    predict_dict = model.predict(preprocessed_images, true_image_shape, **side_inputs)
    predict_dict = ops.bfloat16_to_float32_nested(predict_dict)
    
    # calculate loss
    loss_dict = model.loss(predict_dict, true_image_shape)
    
    # adopt a simpler loss function for now
    # losses = [loss_tensor for loss_tensor in loss_dict.values()]
    losses = [loss_dict['Loss/localization_loss'], loss_dict['Loss/classification_loss']]
    
    # regularization
    if add_regularization_loss:
        regularization_losses = model.regularization_losses()
        
        if regularization_losses:
            regularization_losses = ops.bfloat16_to_float32_nested(regularization_losses)
            regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
            losses.append(regularization_loss)
            loss_dict['Loss/regularization_loss'] = regularization_loss
    
    total_loss = tf.add_n(losses, name='total_loss')
    loss_dict['Loss/total_loss'] = total_loss
    
    return predict_dict, loss_dict
