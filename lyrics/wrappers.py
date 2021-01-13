from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np

from lyrics.world import current_world as world
# Needed to work with keras functional API
virtual_input_layer = tf.keras.Input(shape=(1), dtype=tf.float32)


class VariableWrapperLayer(Layer):
    def __init__(self, tensor, **kwargs):
        super(VariableWrapperLayer, self).__init__(**kwargs)
        self.tensor = tensor

    def call(self, inputs):
        # inputs[0] is a dummy 0, used in order to construct the functional model
        return self.tensor + inputs[0]


class PredicateWrapperLayer(Layer):
    def __init__(self, concrete_function, constraint_shape, atom_shape, **kwargs):
        super(PredicateWrapperLayer, self).__init__(**kwargs)
        self.concrete_function = concrete_function
        self.constraint_shape = np.array(constraint_shape)
        self.atom_shape = np.array(atom_shape)

    def _tile(self, arg_input):
        shape_no_features = np.array(arg_input.shape[:-1])
        shape_no_features = self.atom_shape - shape_no_features + 1
        shape_no_features = np.concatenate((shape_no_features, [1]))
        tiled = tf.tile(arg_input, shape_no_features)
        arg_input = tf.reshape(tiled, (-1, arg_input.shape[-1]))
        return arg_input

    def _back_to_constraint_shape(self, output):
        output = tf.reshape(output, self.atom_shape)
        tile_to_constaint = self.constraint_shape - self.atom_shape + 1
        output = tf.tile(output, tile_to_constaint)
        return output

    def call(self, inputs):
        tensors = []
        for arg_input in inputs:
            tensors.append(self._tile(arg_input))
        ret = self.concrete_function(*tensors)
        ret = self._back_to_constraint_shape(ret)
        return ret


class NotWrapperLayer(Layer):
    def __init__(self, **kwargs):
        super(NotWrapperLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return world.logic.negation(inputs)


class AndWrapperLayer(Layer):
    def __init__(self, **kwargs):
        super(AndWrapperLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return world.logic.weak_conj(inputs)


class OrWrapperLayer(Layer):
    def __init__(self, **kwargs):
        super(OrWrapperLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return world.logic.strong_disj(inputs)


class IffWrapperLayer(Layer):
    def __init__(self, **kwargs):
        super(IffWrapperLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return world.logic.iff(inputs[0], inputs[1])


class ImpliesWrapperLayer(Layer):
    def __init__(self, **kwargs):
        super(ImpliesWrapperLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return world.logic.implication(inputs[0], inputs[1])


class ForAllWrapperLayer(Layer):
    def __init__(self, reduction_axis, **kwargs):
        super(ForAllWrapperLayer, self).__init__(**kwargs)
        self.reduction_axis = reduction_axis

    def call(self, inputs):
        return world.logic.forall(inputs, self.reduction_axis)


class ExistsWrapperLayer(Layer):
    def __init__(self, reduction_axis, **kwargs):
        super(ExistsWrapperLayer, self).__init__(**kwargs)
        self.reduction_axis = reduction_axis

    def call(self, inputs):
        return world.logic.exists(inputs, self.reduction_axis)
