from random import getstate
import numpy as np
from tqdm.auto import tqdm
from scipy.integrate import solve_ivp
from natural_units import fr, to
import dill as pickle
from scipy.integrate._ivp.ivp import OdeResult
from analysis import *

data_folder = "Data/"

def str_fixed(arr):
    return np.array2string(arr, precision=3, sign="-", suppress_small=True, separator=",", floatmode="maxprec_equal").replace(" ", "")

def radiation_reaction_function(u, F, pF):
    wdot = e/m * (np.einsum("l,luv,v", u, pF, u) + e/m * np.einsum("uv,vl,l", F, F, u))
    return 2/3 * e * (np.einsum("u,v->uv", wdot, contract(u)) - np.einsum("u,v->uv", u, contract(wdot)))
radiation_reaction_fun = np.vectorize(radiation_reaction_function, signature="(4),(4,4),(4,4,4)->(4,4)")

def radiation_reaction(u, x, field_tensor, use_radiation_reaction):
    F = field_tensor(x)
    if use_radiation_reaction:
        pF = partial_vector(field_tensor, x, F)
        return F, radiation_reaction_fun(u, F, pF)
    return F, 0*F


def simulation_function(progress, simulation, time_unit, energy_unit, distance_unit):
    def function(tau, y):
        u, x = np.split(y, 2, axis=-1)
        F, F_rad = radiation_reaction(u, x, simulation.field_tensor, simulation.use_radiation_reaction)
        dudtau = e/m * np.einsum("uv,v->u", F + F_rad, u)
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
    return np.vectorize(function, excluded=("tau"), signature="(),(8)->(8)")

class Simulation(dict):
    def __init__(self, tau_span, u0, x0, E_x, B_x, use_radiation_reaction=True, **options):
        self.tau_span=tau_span
        self.u0 = u0
        self.x0 = x0
        self.E_x = E_x
        self.B_x = B_x
        self.field_tensor = field_tensor_functions(self.E_x, self.B_x)[1]
        self.use_radiation_reaction = use_radiation_reaction
        self.options = options
        self.status = -2
    def __call__(self, time_unit="ps", energy_unit="MeV", distance_unit="m", num_points=10000, bar_update_interval=0.1, file="solution"):
        tau_eval = np.linspace(*self.tau_span, num_points)
        y0 = np.concatenate([self.u0, self.x0])
        bar_format = "{l_bar}{bar}| {n_fmt:.5}/{total_fmt:.5}{unit} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        bar_start = to(time_unit, self.tau_span[0])
        bar_end = to(time_unit, self.tau_span[1])
        with tqdm(
            total=bar_end,
            position=bar_start, 
            leave=True, 
            mininterval=bar_update_interval, 
            maxinterval=bar_update_interval,
            bar_format=bar_format,
            unit=time_unit
        ) as progress:
            fun = simulation_function(progress, self, time_unit, energy_unit, distance_unit)
            solution = solve_ivp(
                fun,
                self.tau_span,
                y0,
                t_eval=tau_eval,
                dense_output=True,
                vectorized=False,
                **self.options
            )
            self.sol = solution.sol
            self.nfev = solution.nfev
            self.status = solution.status
            self.message = solution.message
            return fun, solution
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__ = state
    def __repr__(self):
        return self.__dict__.__repr__()
    def save(self, filename):
        with open(data_folder+filename, 'wb+') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
def load_simulation(filename):
    with open(data_folder+filename, 'rb') as handle:
        return pickle.load(handle)