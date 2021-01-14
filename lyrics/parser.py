import tensorflow as tf
import numpy as np

from pyparsing import (
    Suppress,
    Word,
    alphas,
    Forward,
    Keyword,
    Group,
    oneOf,
    delimitedList,
    infixNotation,
    opAssoc,
)

from lyrics.world import current_world as world

from lyrics import wrappers
# Used to return a Variable or create it if it does not exist
from lyrics.wrappers import virtual_input_layer


def create_or_get_variable(tokens, constraint):
    var_name = tokens[0]
    if var_name in constraint.variables_dict:
        return constraint.variables_dict[var_name]
    else:
        new_var = Variable(var_name, constraint)
        constraint.variables_dict[var_name] = new_var
        constraint.variable_indices[var_name] = len(constraint.variables_list)
        constraint.variables_list.append(new_var)
        return new_var


# Most general node class in the logical expression tree
class Node(object):
    def __init__(self):
        self.args = []

# A term can be a constant, a variable or f(t1,t2,...,tn) where f is a function and ti is a term


class Term(Node):
    def __init__(self):
        super(Term, self).__init__()


# This class representes a variable in a logical constraint
class Variable(Term):

    def __init__(self, label, constraint):
        super(Variable, self).__init__()
        self.constraint = constraint
        self.domain = None
        self.tensor = None
        self.layer = None
        self.label = label
        # self.vars is a generic property of Terms that are variable dependent; needed to keep a standard way to process
        self.vars = [self]

    def check_or_assign_domain(self, domain):
        """Variables do not know their domain until a function or a predicate assign one to them on the basis
        of their domain."""

        if self.domain is None:
            self.domain = domain
            self.constraint.var_to_domain[self] = self.domain
        else:
            assert self.domain == domain, "Inconsistency between the domains in which variable %s has been used. Previous: %s, New: %s" %\
                (self.label, self.domain.domain_name, domain.domain_name)

    def compile(self):
        assert self.domain != None, "Trying to compile variable %s before assigning a domain" % self.label
        self.domain_name = self.domain.domain_name
        self.tensor = self.domain.tensor
        # Expanding the variable to the constraint shape.
        for i, var in enumerate(self.constraint.variables_list):
            if var != self:
                self.tensor = tf.expand_dims(self.tensor, i)
        self.layer = wrappers.VariableWrapperLayer(
            tensor=self.tensor)(virtual_input_layer)

        return self.layer


class Atome(Node):

    def __init__(self, tokens, constraint):
        self.constraint = constraint
        self.label = tokens[0]
        if self.label not in world.predicates:
            raise Exception("There is no predicate " + self.label)

        super(Atome, self).__init__()

        self.args = tokens[1:]
        self.predicate = world.predicates[self.label]
        self.vars = []
        assert len(self.args) == len(
            self.predicate.domains), "Wrong number of arguments for predicate " + self.label

        for i, arg in enumerate(self.args):
            assert isinstance(
                arg, Term), "Atomic object %s has an argument that is not a Term" % self.label

            for var in arg.vars:
                if var not in self.vars:
                    self.vars.append(var)
            arg.check_or_assign_domain(self.predicate.domains[i])

    def compile(self):
        self.atom_shape = self.constraint.cartesian_shape.copy()
        for i, var in enumerate(self.constraint.variables_list):
            if var not in self.vars:
                self.atom_shape[i] = 1
        compiled_args = [arg.compile() for arg in self.args]
        predicate_layer = wrappers.PredicateWrapperLayer(concrete_function=self.predicate.concrete_function,
                                                         constraint_shape=self.constraint.cartesian_shape, atom_shape=self.atom_shape)(compiled_args)
        return predicate_layer


class Op(Node):
    def __init__(self, tokens):
        super(Op, self).__init__()
        self.args = tokens[0][0::2]


class Not(object):
    def __init__(self, tokens):
        super(Not, self).__init__()
        self.args = [tokens[0][1]]

    def compile(self):
        assert len(self.args) == 1
        to_negate = self.args[0].compile()
        return wrappers.NotWrapperLayer()(to_negate)


class And(Op):

    def __init__(self, tokens):
        super(And, self).__init__(tokens)

    def compile(self):
        compiled_args = [arg.compile() for arg in self.args]
        return wrappers.AndWrapperLayer()(compiled_args)


class Or(Op):

    def __init__(self, tokens):
        super(Or, self).__init__(tokens)

    def compile(self):
        compiled_args = [arg.compile() for arg in self.args]
        return wrappers.OrWrapperLayer(name='Or')(compiled_args)


class Iff(Op):

    def __init__(self, tokens):
        super(Iff, self).__init__(tokens)

    def compile(self):
        assert len(
            self.args) == 2, "n-ary double implication not allowed. Use parentheses to group chains of implications"
        left = self.args[0].compile()
        right = self.args[1].compile()

        return wrappers.IffWrapperLayer(name='equivalence')([left, right])


class Implies(Op):

    def __init__(self, tokens):
        super(Implies, self).__init__(tokens)

    def compile(self):
        assert len(
            self.args) == 2, "n-ary implication not allowed. Use parentheses to group chains of implications"
        left = self.args[0].compile()
        right = self.args[1].compile()
        return wrappers.ImpliesWrapperLayer(name='implies')([left, right])


class Quantifier(Node):
    def __init__(self, tokens):
        super(Quantifier, self).__init__()
        self.args = [tokens[2][0]]
        self.variable = tokens[1]

    def __str__(self):
        return self.label + " " + self.var


class ForAll(Quantifier):

    def __init__(self, constraint, tokens):
        super(ForAll, self).__init__(tokens)
        self.constraint = constraint

    def compile(self):
        variable_axis = self.constraint.variable_indices[self.variable.label]
        compiled = self.args[0].compile()
        return wrappers.ForAllWrapperLayer(reduction_axis=variable_axis)(compiled)


class Exists(Quantifier):

    def __init__(self, constraint, tokens):
        super(Exists, self).__init__(tokens)
        self.constraint = constraint

    def compile(self):
        variable_axis = self.constraint.variable_indices[self.variable.label]
        compiled = self.args[0].compile()
        return wrappers.ExistsWrapperLayer(reduction_axis=variable_axis)(compiled)


class FOLParser(object):
    def _createParseAction(self, class_name, constraint):
        def _create(tokens):
            if class_name == "Variable":
                return create_or_get_variable(tokens, constraint)
            elif class_name == "Atome":
                return Atome(tokens, constraint)
            elif class_name == "IMPLIES":
                return Implies(tokens)
            elif class_name == "IFF":
                return Iff(tokens)
            elif class_name == "FORALL":
                return ForAll(constraint, tokens)
            elif class_name == "EXISTS":
                return Exists(constraint, tokens)
            elif class_name == "OR":
                return Or(tokens)
            elif class_name == "AND":
                return And(tokens)
            elif class_name == "NOT":
                return Not(tokens)
        return _create

    def parse(self, constraint):

        left_parenthesis, right_parenthesis, colon, left_square, right_square = map(
            Suppress, '():[]')
        symbol = Word(alphas)

        variable = symbol
        variable.setParseAction(
            self._createParseAction("Variable", constraint))

        term = variable
        predicate = oneOf(list(world.predicates.keys()))

        atomic_formula = predicate + left_parenthesis + \
            delimitedList(term) + right_parenthesis
        atomic_formula.setParseAction(
            self._createParseAction("Atome", constraint))

        implies = Keyword("->")
        iff = Keyword("<->")
        not_ = Keyword("not")
        and_ = Keyword("and")
        or_ = Keyword("or")

        atomic_forumulas = infixNotation(atomic_formula, [
            (not_, 1, opAssoc.RIGHT, self._createParseAction("NOT", constraint)),
            (and_, 2, opAssoc.LEFT, self._createParseAction("AND", constraint)),
            (or_, 2, opAssoc.LEFT, self._createParseAction("OR", constraint)),
            (implies, 2, opAssoc.RIGHT, self._createParseAction("IMPLIES", constraint)),
            (iff, 2, opAssoc.RIGHT, self._createParseAction("IFF", constraint)),
        ])

        forall_quantifier = Keyword("forall") + symbol + colon
        exists_quantifier = Keyword("exists") + symbol + colon

        formula = Forward()
        forall_formula = forall_quantifier + \
            Group(atomic_forumulas) | forall_quantifier + Group(formula)
        exists_formula = exists_quantifier + \
            Group(atomic_forumulas) | exists_quantifier + Group(formula)
        formula <<= forall_formula | exists_formula

        forall_formula.setParseAction(
            self._createParseAction("FORALL", constraint))
        exists_formula.setParseAction(
            self._createParseAction("EXISTS", constraint))

        tree = formula.parseString(constraint.definition, parseAll=True)
        return tree[0]
