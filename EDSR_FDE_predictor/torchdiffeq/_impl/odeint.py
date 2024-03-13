from .tsit5 import Tsit5Solver
from .dopri5 import Dopri5Solver
from .bosh3 import Bosh3Solver
from .adaptive_heun import AdaptiveHeunSolver
from .fixed_grid import Euler, Midpoint, RK4
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .adams import VariableCoefficientAdamsBashforth
from .misc import _check_inputs
'''
这里 AdamsBashforth、AdamsBashforthMoulton、Euler、Midpoint、RK4 
(Fourth-order Runge-Kutta with 3/8 rule) 属于 FixedGridODESolver (固定网格 ODE 求解器)，其中，
前两个 Adams 类型的求解器 是作者自己实现的 Adam梯度下降方法来求解的 FixedGridODESolver。
而VariableCoefficientAdamsBashforth、Tsit5Solver （）、Dopri5Solver （Runge-Kutta 4(5)）属于
 AdaptiveStepsizeODESolver（自定义步长的 ODE 求解器）。
'''
SOLVERS = {
    'explicit_adams': AdamsBashforth,
    'fixed_adams': AdamsBashforthMoulton,
    'adams': VariableCoefficientAdamsBashforth,
    'tsit5': Tsit5Solver,
    'dopri5': Dopri5Solver,
    'bosh3': Bosh3Solver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
    'adaptive_heun': AdaptiveHeunSolver,
}


def odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

   Args:
        func: Function that maps a Tensor holding the state `y` and a scalar Tensor
            `t` into a Tensor of state derivatives with respect to time.
    func：把一个含有状态张量 y 和常张量 t 映射到 一个关于时间可导的张量上。
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. May
            have any floating point or complex dtype.
    y0: NxD维度的张量，是 y 在 t[0] 的初始点，可以是任意复杂的类型。
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`. The initial time point should be the first element of this sequence,
            and each time must be larger than the previous time. May have any floating
            point dtype. Converted to a Tensor with float64 dtype.
    t: 1xD的张量，表示一系列用于求解 y 的时间点。
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
    rtol: 相对错误容忍度，以限制张量 y 中每个元素的上限值。（可调节）
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
    atol: 绝对错误容忍度，以限制张量 y 中每个元素的上限值。（可调节）
        method: optional string indicating the integration method to use.
        method: 可选的string型 以决定那种 积分方法 被使用。
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
    options: 可选的字典类型，用于配置积分方法。
        name: Optional name for this operation.
    name:  为该操作指定名称。
    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.
    Returns: 返回第一个维度对应不同的时间点的 y 张量。
             包含 y 在每个时间点 t 上被期望的解。（所有时间点的解都被求得了），
             初始值 y0 是第一维度的第一个元素。


    Raises:
        ValueError: if an invalid `method` is provided.
        TypeError: if `options` is supplied without `method`, or if `t` or `y0` has
            an invalid dtype.
    """

    tensor_input, func, y0, t = _check_inputs(func, y0, t)

    if options is None:
        options = {}
    elif method is None:
        raise ValueError('cannot supply `options` without specifying `method`')

    if method is None:
        method = 'dopri5'

    solver = SOLVERS[method](func, y0, rtol=rtol, atol=atol, **options)
    solution = solver.integrate(t)

    if tensor_input:
        solution = solution[0]
    return solution
