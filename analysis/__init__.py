import numpy as np
from natural_units import fr, to
from scipy.special import ellipk, ellipe, kv, fresnel

e = fr("C", -1.602176634e-19)
c = 1
m = fr("eV", 5.1099906e5)

levi_civita = np.zeros((3,3,3))
levi_civita[0,1,2] = levi_civita[1,2,0] = levi_civita[2,0,1] = 1
levi_civita[2,1,0] = levi_civita[1,0,2] = levi_civita[0,2,1] = -1

def magnetic_coil_vector(B_0, a, pos):
    r = np.linalg.norm(pos[1:2])
    x = np.abs(pos[0])
    if r == 0:
        if x == 0:
            return B_0
        return B_0 * a**3 / (x**2 + a**2)**1.5
    alpha = r/a
    beta = x/a
    gamma = x/r
    Q = (1 + alpha)**2 + beta**2
    k = np.sqrt(4 * alpha / Q)
    K = ellipk(k**2)
    E = ellipe(k**2)
    Bx = np.linalg.norm(B_0) / (np.pi * np.sqrt(Q)) * (E * (1 - alpha**2 - beta**2)/(Q - 4*alpha) + K)
    Br = np.linalg.norm(B_0) * gamma / (np.pi * np.sqrt(Q)) * (E * (1 + alpha**2 + beta**2)/(Q - 4*alpha) - K)
    return Bx * np.array([1, 0, 0]) + Br * np.array([0, pos[1], pos[2]])/r

def partial_scalar(original, function, x, direction, delta=1e-10):
    return (function(x + direction*delta) - original)/delta

def partial_vector(f, x, original=None):
    if original is None:
        original = f(x)
    return np.apply_along_axis(lambda direction: partial_scalar(original, f, x, direction), 0, np.eye(4))

def contract(X, axis=-1, eta = np.diag([1, -1, -1, -1])):
    return np.swapaxes(np.tensordot(eta, X, (1, axis)), 0, axis)

def gamma_p(p, axis=-1):
    return np.sqrt(1 + (np.linalg.norm(np.take(p, (1, 2, 3), axis=axis), axis=axis)/m/c)**2)

def gamma_u(u, axis=-1):
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
        return contract(contravariant_field_tensor(x), 0)
    def covariant_field_tensor(x):
        return contract(mixed_field_tensor(x), 1)
    functions = contravariant_field_tensor, mixed_field_tensor, covariant_field_tensor
    return [np.vectorize(function, signature="(4)->(4,4)") for function in functions]

def contravariant_field_generated(fun, solution, obs_tau):
    def contravariant_field_generated_function(tau):
        y = solution.sol(tau)
        u, x = np.split(y, 2)
        tau_0 = tau - np.linalg.norm(x - obs_tau(tau), axis=0)/c
        r = obs_tau(tau_0)
        dydtau = fun(tau_0, y)
        dudtau, dxdtau = np.split(dydtau, 2)
        a = dudtau
        xminr = x - r
        lowered_xminr = contract(xminr)
        xminrdotu = np.einsum("l,l", lowered_xminr, u)
        if xminrdotu == 0:
            return np.zeros((4,4))
        xminrdota = np.einsum("l,l", lowered_xminr, a)
        xminrouta = np.einsum("a,b", xminr, a)
        xminroutv = np.einsum("a,b", xminr, u)
        return e/xminrdotu**3 * (xminrdotu * (xminrouta - xminrouta.T) - (c**2 + xminrdota) *  (xminroutv - xminroutv.T))
    return np.vectorize(contravariant_field_generated_function, signature="()->(4,4)")

def contravariant_stress_energy_function(contravariant_F):
    mixed_F = contract(contravariant_F, 0)
    covariant_F = contract(mixed_F, 1)
    return np.einsum("ua,av->uv", contravariant_F, mixed_F) - 1/4 * contract(np.eye(4)) * np.einsum("ab,ab", covariant_F, contravariant_F)
contravariant_stress_energy = np.vectorize(contravariant_stress_energy_function, signature="(4,4)->(4,4)")

def electric_field_function(contravariant_F):
    return 1/2 * (contravariant_F[0,1:] - contravariant_F[1:,0].T)
electric_vector = np.vectorize(electric_field_function, signature="(4,4)->(3)")

def magnetic_field_function(contravariant_F):
    return - 1/2 * np.einsum("ijk,jk->i", levi_civita, contravariant_F[1:,1:])
magnetic_vector = np.vectorize(magnetic_field_function, signature="(4,4)->(3)")

def poynting_vector_function(contravariant_T):
    return c/2 * (contravariant_T[0,1:] - contravariant_T[1:,0].T)
poynting_vector = np.vectorize(poynting_vector_function, signature="(4,4)->(3)")

def omega_function(r_corr, v_corr):
    radius = np.linalg.norm(r_corr[1:])
    if radius == 0:
        return np.zeros(3)
    return np.cross(r_corr, v_corr)/radius**2
omega_vector = np.vectorize(omega_function, signature="(3),(3)->(3)")

def contravariant_angular_momentum_function(x, p):
    return np.einsum("a,b->ab", x, p) - np.einsum("b,a->ab", x, p)
contravariant_angular_momentum = np.vectorize(contravariant_angular_momentum_function, signature="(4),(4)->(4,4)")

def analytic_spectral_intensity_function(omega, theta, gamma, rho):
    xi = omega * rho / (3 * c * gamma**3) / (1 + theta**2 * gamma**2)**1.5
    return e**2 * c / (12 * np.pi**3) * omega**2 * (rho/c)**2 * (gamma**-2 + theta**2)**2 * (kv([2/3], [xi])[0]**2 + theta**2 / (1/gamma**2 + theta**2) * kv([1/3], [xi])[0]**2)
analytic_spectral_intensity = np.vectorize(analytic_spectral_intensity_function, excluded=["theta", "gamma", "rho"])

def masked(data, mask):
    return np.compress(mask, data, axis=0)

def taylor_intensity(Dtau, u0n, u1n, chi1n, chi2n, sgn):
    I0n = np.sinc(chi1n * Dtau / 2) * Dtau
    I1n = 1 / chi1n * (I0n - np.cos(chi1n * Dtau / 2) * Dtau)
    I2n = Dtau**2 / 4 * I0n - 2 / chi1n * I1n
    
    ReIn = u0n * I0n[:, None]
    ImIn = u1n * I1n[:, None] + sgn * chi2n[:, None] * u0n * I2n[:, None]
    return ReIn, ImIn

def fresnel_intensity(Dtau, u0n, u1n, chi1n, chi2n, sgn):
    Theta_b = np.sqrt(2 * np.pi * chi2n)
    Theta_plus = (chi1n + chi2n*Dtau)/Theta_b
    Theta_minus = (chi1n - chi2n*Dtau)/Theta_b

    Psi_a = np.sqrt(2 * np.pi / chi2n)[:, None] * (2 * chi2n[:, None] * u0n - chi1n[:, None] * u1n)
    Psi_b = chi1n[:, None]**2 / (4 * chi2n[:, None])
    Psi_plus = Psi_a * np.cos(Psi_b)
    Psi_minus = Psi_a * np.sin(Psi_b)

    Phi_a = Dtau[:, None]**2/4 * chi2n[:, None]
    Phi_b = Dtau[:, None]/2 * chi1n[:, None]
    Phi_plus = Phi_a + Phi_b
    Phi_minus = Phi_a - Phi_b

    S_plus, C_plus = fresnel(Theta_plus)
    S_minus, C_minus = fresnel(Theta_minus)
    C = C_plus[:, None] - C_minus[:, None]
    S = S_plus[:, None] - S_minus[:, None]

    ReIn = 1/(4*chi2n[:, None]) * (Psi_plus * C + Psi_minus * S + 2 * u1n * (np.sin(Phi_plus) - np.sin(Phi_minus)))
    ImIn = sgn * 1/(4*chi2n[:, None]) * (Psi_plus * S - Psi_minus * C - 2 * u1n * (np.cos(Phi_plus) - np.cos(Phi_minus)))
    return ReIn, ImIn


def spectral_intensity(sim, obs_tau, T=1e-3):
    def spectral_intensity_function(omega, theta):
        kappa = omega*np.concatenate([np.ones([shat.shape[0], 1]), shat/c], axis=1)
        chi0n = np.einsum("...a,...a->...", contract(kappa), x0n)
        chi1n = np.einsum("...a,...a->...", contract(kappa), x1n)
        chi2n = np.einsum("...a,...a->...", contract(kappa), x2n)
        sgn = np.sign(omega)
        Tval = chi2n * Dtau**2
        mask = Tval <= T
        # use Taylor series
        Taylor_ReIn, Taylor_ImIn = taylor_intensity(masked(Dtau, mask), masked(u0n, mask), masked(u1n, mask), masked(chi1n, mask), masked(chi2n, mask), sgn)
        # use Fresnel integrals
        Fresnel_ReIn, Fresnel_ImIn = fresnel_intensity(masked(Dtau, ~mask), masked(u0n, ~mask), masked(u1n, ~mask), masked(chi1n, ~mask), masked(chi2n, ~mask), sgn)
        mask = np.repeat(mask[:, None], 4, 1)
        ReIn = np.zeros(x0n.shape)
        np.place(ReIn, mask, Taylor_ReIn.flatten())
        np.place(ReIn, ~mask, Fresnel_ReIn.flatten())
        ImIn = np.zeros(x0n.shape)
        np.place(ImIn, mask, Taylor_ImIn.flatten())
        np.place(ImIn, ~mask, Fresnel_ImIn.flatten())
        ReS = np.sum(ReIn * np.cos(chi0n)[:, None] - ImIn * np.sin(chi0n)[:, None], axis=0)[1:]
        ImS = np.sum(ImIn * np.cos(chi0n)[:, None] - ReIn * np.sin(chi0n)[:, None], axis=0)[1:]
        dIdomegadOmega = e**2 * c / (16 * np.pi**3) * omega**2 * (ReS[0]**2 + ImS[0]**2 + (ReS[1]*np.cos(theta) - ReS[2]*np.sin(theta))**2 + (ImS[1]*np.cos(theta) - ImS[2]*np.sin(theta))**2)
        return dIdomegadOmega

    interpolants = np.array(list(map(lambda interpolant: interpolant.Q, sim.sol.interpolants)))
    tau = sim.sol.ts[1:]
    Dtau = np.diff(sim.sol.ts)
    y = sim.sol(tau).T
    u, x = np.split(y, 2, axis=-1)
    xo = obs_tau(tau)
    shat = xo - x
    shat = shat[:,1:]
    shat /= np.linalg.norm(shat, axis=1)[:, None]
    u0n, u1n = np.moveaxis(interpolants[:,:4,:2], 2, 0)
    x0n, x1n, x2n = np.moveaxis(interpolants[:,4:,:3], 2, 0)
    return np.vectorize(spectral_intensity_function, excluded=["theta"])