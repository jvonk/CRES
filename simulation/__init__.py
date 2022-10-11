import numpy as np
from tqdm.auto import tqdm
from scipy.integrate import solve_ivp
from natural_units import fr, to
import pickle
from scipy.integrate._ivp.ivp import OdeResult

e = fr("C", 1.602176634e-19)
c = 1
m = fr("eV", 5.1099906e5)

def partial_scalar(original, function, x, direction, delta=1e-10):
    return (function(x + direction*delta) - original)/delta

def partial_vector(f, x, original=None):
    if original is None:
        original = f(x)
    return np.apply_along_axis(lambda direction: partial_scalar(original, f, x, direction), 0, np.eye(4))

def contract(X, eta = np.diag([1, -1, -1, -1])):
    return np.einsum("ab,b...->a...", eta, X)

def gamma_p(p, axis=0):
    return np.sqrt(1 + (np.linalg.norm(p[1:], axis=axis)/m/c)**2)

def gamma_u(u, axis=0):
    return gamma_p(u*m, axis=axis)

def field_tensor_functions(E_x, B_x):
    def contravariant_field_tensor(x):
        Ex, Ey, Ez = E_x(x)
        Bx, By, Bz = B_x(x)
        return np.array([
            [0, -Ex, -Ey, -Ez],
            [Ex, 0, -Bz, By],
            [Ey, Bz, 0, -Bx],
            [Ez, -By, Bx, 0]
        ])
    def mixed_field_tensor(x):
        return contract(contravariant_field_tensor(x))
    def covariant_field_tensor(x):
        return contract(mixed_field_tensor(x))
    return contravariant_field_tensor, mixed_field_tensor, covariant_field_tensor

def str_fixed(arr):
    return np.array2string(arr, precision=3, sign="-", suppress_small=True, separator=",", floatmode="maxprec_equal").replace(" ", "")

def simulation_function(progress, E_x, B_x, time_unit, energy_unit, distance_unit):
    def function(tau, y):
        field_tensor = field_tensor_functions(E_x, B_x)[1]
        u, x = np.split(y, 2)
        F = field_tensor(x)
        pF = partial_vector(field_tensor, x, F)
        wdot = e/m * (np.einsum("l,luv,v", u, pF, u) + e/m * np.einsum("uv,vl,l", F, F, u))
        dudtau = e/m * np.einsum("uv,v->u", F, u)
        dudtau += 2/3 * e**2 / m * np.einsum("uv,v->u", np.einsum("u,v->uv", wdot, contract(u)) - np.einsum("u,v->uv", u, contract(wdot)), u)
        dxdtau = u
        dydtau = np.concatenate([dudtau, dxdtau])
        progress.set_postfix(
            p = f"{str_fixed(to(energy_unit, u*m))}{energy_unit}",
            t = f"{str_fixed(to(time_unit, x[0]))}{time_unit}",
            r = f"{str_fixed(to(distance_unit, x[1:]))}{distance_unit}",
            refresh = False
        )
        progress.n = to(time_unit, tau)
        progress.update(0)
        return dydtau
    return function

def simulate(tau_span, u0, x0, E_x, B_x, time_unit="ps", energy_unit="MeV", distance_unit="m", num_points=10000, bar_update_interval=0.1, **options):
    tau_eval = np.linspace(*tau_span, num_points)
    y0 = np.concatenate([u0, x0])
    bar_format = "{l_bar}{bar}| {n_fmt:.5}/{total_fmt:.5}{unit} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    bar_start = to(time_unit, tau_span[0])
    bar_end = to(time_unit, tau_span[1])
    with tqdm(
            total=bar_end,
            position=bar_start, 
            leave=True, 
            mininterval=bar_update_interval, 
            maxinterval=bar_update_interval,
            bar_format=bar_format,
            unit=time_unit
        ) as progress:
        fun = simulation_function(progress, E_x, B_x, time_unit, energy_unit, distance_unit)
        return fun, solve_ivp(
            fun,
            tau_span,
            y0,
            t_eval=tau_eval,
            dense_output=True,
            **options
        )

def save_solution(file, solution):
    np.savez(file, **dict(solution))
    with open(file, 'wb') as handle:
        pickle.dump(dict(solution), handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_solution(file):
    with open(file, 'rb') as handle:
        solution_dict = pickle.load(handle)
        return OdeResult(solution_dict)