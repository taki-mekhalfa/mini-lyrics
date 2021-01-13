import abc
import tensorflow as tf


class FuzzyLogic(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def weak_conj(args):
        raise NotImplementedError(
            'users must define "weak_conj" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def strong_disj(args):
        raise NotImplementedError(
            'users must define "strong_disj" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def exclusive_disj(args):
        raise NotImplementedError(
            'users must define "exclusive_disj" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def forall(a, axis):
        raise NotImplementedError(
            'users must define "forall" to use this base class')

    @classmethod
    def forall_with_loss(cls, a, axis):
        return cls.forall(a, axis)

    @staticmethod
    @abc.abstractmethod
    def exists(a, axis):
        raise NotImplementedError(
            'users must define "exists" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def exists_n(a, axis, n):
        raise NotImplementedError(
            'users must define "exists_n" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def negation(a):
        raise NotImplementedError(
            'users must define "negation" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def implication(a, b):
        raise NotImplementedError(
            'users must define "implies" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def iff(a, b):
        raise NotImplementedError(
            'users must define "iff" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def loss(a):
        raise NotImplementedError(
            'users must define "loss" to use this base class')


class Lukasiewicz(FuzzyLogic):

    @staticmethod
    def weak_conj(args):
        return tf.reduce_min(args, axis=0)
    
    @staticmethod
    def strong_disj(args):
        return tf.minimum(1., tf.reduce_sum(args, axis=0))

    @staticmethod
    def forall(a, axis=0):
        return tf.reduce_min(a, axis=axis)

    @staticmethod
    def exists(a, axis):
        return tf.reduce_max(a, axis=axis)

    @staticmethod
    def negation(a):
        return 1. - a

    @staticmethod
    def implication(a, b):
        return tf.minimum(1., 1 - a + b)

    @staticmethod
    def iff(a, b):
        return 1 - tf.abs(a-b)

    @staticmethod
    def loss(phi):
        return 1. - phi

class LogicFactory:

    @staticmethod
    def create(logic):
        if logic == "lukasiewicz":
            return Lukasiewicz
        else:
            raise Exception("Logic %s unknown" % logic)
