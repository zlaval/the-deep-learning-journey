import math

import numpy as np
from matplotlib import pyplot as plt
from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value: {self.data}"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad = (1 - t ** 2) * out.grad
            print(self.grad)

        out._backward = _backward
        return out


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

def test():
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    b=Value(6.8813735870195432, label='b')

    x1w1=x1*w1
    x1w1.label = 'x1w1'

    x2w2=x2*w2
    x2w2.label = 'x2w2'

    x1w1x2w2 = x1w1+x2w2
    x1w1x2w2.label = 'x1w1x2w2'

    n=x1w1x2w2+b
    n.label = 'n'

    o=n.tanh()
    o.grad = 1.0
    o.label = 'o'

    o._backward()
    n._backward()
    b._backward()
    x1w1x2w2._backward()
    x1w1._backward()
    x2w2._backward()

    dot = draw_dot(o)
    dot.render("graph_output", view=True)


if __name__ == '__main__':
    test()


def value_playground():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f
    L.label = 'L'

    print(L)

    # e = a*b
    # d = e+c
    # L = (a*b+c)*f

    # print(d)
    # print(d._prev)
    # print(d._op)
    dot = draw_dot(L)
    dot.render("graph_output", view=True)

    # derivative
    # (f(x+h)-f(x)) / h

    # dL/dd = -2 (equals to gradient)

    # dd / dc = ?
    # ((c+h +e) - (c+e)) / h = 1.0

    # chain rule
    # h'(x) = f'(g(x))*g'(x)

    # dL / dc = (dL / dd) * (dd / dc) = -2*1=-2

    # d/dx(tanh x)= 1-tanh**2 x
