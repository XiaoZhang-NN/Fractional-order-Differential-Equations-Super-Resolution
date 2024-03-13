from .solvers import FixedGridODESolver
from . import rk_common

"""
论文中把 ODE solver 当作一个黑盒子（black box），我们知道它可以求解我们所需要的微分方程。这里只看最简单的 Euler 求解器：
它只是实现了父类 FixedGridODESolver 中的 step_func，父类 FixedGridODESolver 的实现为class FixedGridODESolver(object):solvers.py
"""
class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return tuple(dt * f_ for f_ in func(t, y))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return rk_common.rk4_alt_step_func(func, t, dt, y)

    @property
    def order(self):
        return 4
