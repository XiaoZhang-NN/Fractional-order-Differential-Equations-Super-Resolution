import abc
import torch
from .misc import _assert_increasing, _handle_unused_kwargs


class AdaptiveStepsizeODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, atol, rtol, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.atol = atol
        self.rtol = rtol

    def before_integrate(self, t):
        pass

    @abc.abstractmethod
    def advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        _assert_increasing(t)
        solution = [self.y0]
        t = t.to(self.y0[0].device, torch.float64)
        self.before_integrate(t)
        for i in range(1, len(t)):
            y = self.advance(t[i])
            solution.append(y)
        return tuple(map(torch.stack, tuple(zip(*solution))))

    """
     fixed_grid.py::class Euler(FixedGridODESolver)实现了父类 FixedGridODESolver 中的 step_func，父类 FixedGridODESolver 的实现为：
    """
class FixedGridODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, step_size=None, grid_constructor=None, **unused_kwargs):
        """
        here, I omit some initialize progress in origin code and omit some grid constructor progress
        """
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('atol', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0

        if step_size is not None and grid_constructor is None:
            self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        elif grid_constructor is None:
            self.grid_constructor = lambda f, y0, t: t
        else:
            raise ValueError("step_size and grid_constructor are exclusive arguments.")

    def _grid_constructor_from_step_size(self, step_size):

        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters).to(t) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    @property
    @abc.abstractmethod
    def order(self):
        pass

    @abc.abstractmethod
    def step_func(self, func, t, dt, y):
        pass

    def integrate(self, t):
        _assert_increasing(t) 
        # t is increase sequence
        t = t.type_as(self.y0[0])
        time_grid = self.grid_constructor(self.func, self.y0, t)
        # grad

        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        time_grid = time_grid.to(self.y0[0])

        solution = [self.y0]
        # target solution list


        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self.step_func(self.func, t0, t1 - t0, y0)
            # use step function
            y1 = tuple(y0_ + dy_ for y0_, dy_ in zip(y0, dy))
            # y1=y0+dy

            while j < len(t) and t1 >= t[j]:
                solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                j += 1
            y0 = y1
            # why to this? linear interpolate the time sequence

        return tuple(map(torch.stack, tuple(zip(*solution))))

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
        slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
        return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))
        """
        这里的积分 应该是对 差分 的积分，即根据初始值y0和时间序列t求yt。
        首先构建 time grad,然后使用step_ func,根据func(NN中的/)和timegrad中
        的t以及yo来计算dy,接着，根据yn=y0+dy求得y1,这里有一行yo=yn,为什么把y1
        赋值给y0?然后再根据y0, y1求插值?这样元素不就等于零了?
        """
