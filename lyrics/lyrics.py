from lyrics.parser import FOLParser
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from lyrics.fuzzy import LogicFactory
from lyrics.world import current_world as world
from lyrics.wrappers import virtual_input_layer


class Domain(object):

    @staticmethod
    def get_domain(domain):
        if domain in world.domains:
            return world.domains[domain]
        else:
            raise Exception("Domain %s not recognised" % domain_name)

    def __init__(self, domain_name, elements):
        if domain_name in world.domains:
            raise Exception("Domain %s already exists" % domain_name)
        self.domain_name = domain_name

        assert isinstance(elements, tf.Tensor) or isinstance(
            elements, np.ndarray), "Elements for domain %s should be provided as a Tensorflow tensor or a numpy ndarrays" % domain_name

        self.tensor = tf.constant(elements, dtype=tf.float32)
        assert len(self.tensor.shape) == 2, "Data for domain %s must be a two-dimensional tensor(i.e. matrix where rows correspond to individuals and columns to concrete individuals feature representation)" % self.domain_name

        world.domains[self.domain_name] = self

    def __str__(self):
        return str(self.tensor.numpy())


class Function(object):
    def __init__(self, function_name, domains, concrete_function):
        if function_name in world.functions:
            raise Exception("Function %s already exists" % function_name)
        self.function_name = function_name
        self.domains = []

        for domain_name in domains:
            if domain_name in world.domains:
                self.domains.append(world.domains[domain_name])
            else:
                raise Exception("Domaine %s is not known for it to be used with the function %s" % (
                    domain_name, function_name))

        world.functions[function_name] = self
        self.arity = len(self.domains)

        if concrete_function is None:
            raise NotImplementedError(
                "Default concrete functions implementation in function %s hasn't been provided" % function_name)
        else:
            self.concrete_function = concrete_function


class Predicate(object):
    def __init__(self, predicate_name, domains, concrete_function):
        if predicate_name in world.predicates:
            raise Exception("Prdicate %s already exists" % predicate_name)
        self.predicate_name = predicate_name
        self.domains = []
        for domain_name in domains:
            if domain_name in world.domains:
                self.domains.append(world.domains[domain_name])
            else:
                raise Exception("Domaine %s is not known for it to be used with the predicate %s" % (
                    domain_name, predicate_name))

        world.predicates[predicate_name] = self
        self.arity = len(self.domains)

        if concrete_function is None:
            raise NotImplementedError(
                "Default concrete function implementation in predicate %s hasn't been provided" % predicate_name)
        else:
            self.concrete_function = concrete_function


class PointwiseConstraint(object):
    def __init__(self, concrete_function, y, x, weight=1):
        self.concrete_function = concrete_function
        self.x = x
        self.y = y
        self.weight = weight

        world.point_wise_constraints.append(self)

    def loss(self):
        return self.weight * self.concrete_function.cost(self.y, self.x)


class Constraint(object):

    def __init__(self, definition, weight=1):

        self.definition = definition
        self.weight = weight

        # Each constraint will have a set of variables
        # These will be filled by the parser
        self.variables_dict = {}
        self.variables_list = []
        self.variable_indices = {}
        self.var_to_domain = {}

        # Parsing the FOL formula
        parser = FOLParser()

        expr_tree = parser.parse(constraint=self)

        # Compute the cartesian shape of the constraint. Needed for compilation into a model
        # This is the shape of the multi-dimensional tensor, where each dimension corresponds to a variable
        self.cartesian_shape = []

        for var in self.variables_list:
            if var in self.var_to_domain:
                self.cartesian_shape.append(
                    self.var_to_domain[var].tensor.shape[0])
            else:
                raise Exception(
                    "In constraint [%s], a variable has not been used in any predicate or function" % self.definition)

        # Compiling the constraint expression tree.
        compiled = expr_tree.compile()
        self.model = Model(inputs=virtual_input_layer, outputs=compiled)
        world.constraints.append(self)

    def loss(self,):
        # 0 is needed as input for the first layer => to be able to work with tensorflow's functiona API
        return self.weight * world.logic.loss(self.model(np.array([0.])))
