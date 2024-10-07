from keras import Layer
from keras import ops
class AdvancedPartialTransformLayer(Layer):
    def __init__(self, **kwargs):
        super(AdvancedPartialTransformLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, masks, transformations = inputs
        original_shape = ops.shape(x)
        n_transformations = ops.shape(masks)[-1]
        ts = ops.shape(transformations)
        tranformations = ops.reshape((ts[0], ts[1], ts[2], -1, n_transformations , ))


        result = x
        for i in (range(n_transformations)):
            mask = masks[:, :, :, i:i+1]
            transformed = tranformations[:, :, :, :, i]
            expanded_mask = ops.repeat(mask, repeats=original_shape[-1], axis=-1)
            result = result * (1 - expanded_mask) + transformed * expanded_mask

        return result
