import tensorflow as tf

from lyrics.fuzzy import LogicFactory

class World(object):
    def __init__(self, logic="lukasiewicz"):
        # Map from domain name to domain object
        self.domains = {}

        # Map from function name to function object
        self.functions = {}

        # Map from relation name to relation object
        self.predicates = {}

        # A list of pointwise constraints to be added the overall cost
        self.point_wise_constraints = []

        # A list of high level constraints
        self.constraints = []

        # Fuzzy logic utilities
        self.logic = LogicFactory.create(logic)

    def loss(self):
        loss = tf.constant(0.)
        for ptc in self.point_wise_constraints:
            loss += ptc.loss()
        for c in self.constraints:
            loss += c.loss()
        return loss
    


# Represents the current world
current_world = World()
