import numpy as np
from matplotlib import pyplot as plt
from natural_units import fr, to

standard = (8, 6)
square = (7.5, 10)
double_square = (15, 10)
large_standard = (16, 12)

def plot(u, label_u, unit_u, c=None, label_c=None, unit_c=None, fig=plt.figure(), ax=None, position=111, projection="rectilinear"):
    if ax is None:
        ax = fig.add_subplot(position, projection=projection)
    u = to(unit_u, u)
    if c is not None:
        if unit_c is not None:
            c = to(unit_c, c)
        if u.ndim > 1 and u.shape[0] > 1:
            sc = ax.scatter(*u, c=c, cmap="hot", marker=".", s=1)
            ax.set_xlabel(f"{label_u}_x ({unit_u})")
            ax.set_ylabel(f"{label_u}_y ({unit_u})")
            if u.shape[0] > 2:
                ax.set_zlabel(f"{label_u}_z ({unit_u})")
                plt.colorbar(sc, ax=ax, location="bottom", shrink=0.5, pad=0.01, label=f"{label_c} ({unit_c})")
            else:
                plt.colorbar(sc, ax=ax, location="bottom", pad=0.1, label=f"{label_c} ({unit_c})")
        else:
            if u.ndim > 1:
                u = u[0]
            ax.scatter(c, u, marker=".", s=1)
            ax.set_xlabel(f"{label_c} ({unit_c})")
            ax.set_ylabel(f"{label_u} ({unit_u})")
    else:
        if u.ndim > 1 and u.shape[0] > 1:
            sc = ax.scatter(*u, marker=".", s=1)
            ax.set_xlabel(f"{label_u}_x ({unit_u})")
            ax.set_ylabel(f"{label_u}_y ({unit_u})")
            if u.shape[0] > 2:
                ax.set_zlabel(f"{label_u}_z ({unit_u})")
    return ax

def linear_fit(x, y, x_label="x", y_label="y", units=""):
    fit, cov = np.polyfit(x, y, 1, cov=True)
    uncertainty = np.sqrt(np.diag(cov))
    f = np.poly1d(fit)
    print(f"{y_label} = {fit[0]:.3}±{uncertainty[0]:.3} {x_label} + {fit[1]:.3}±{uncertainty[1]:.3} {units}")
    return f