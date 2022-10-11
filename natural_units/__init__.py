import re
import numpy as np

# Using natural units
# eps_0 = 1
# mu_0 = 1
# c = 1
# hbar = 1
# k_B = 1
# e = 1
# In SI units,
# c = 2.99792458e8 m/s
# hbar = 6.62607015e-34/(2*pi)
# k_B = 1.380649e-23 J/K
# mu_0 = 1.25663706212e-6 J/m/A^2 (experimental)
# 1 eV = 1.602176634e-19 J
# e = 1.602176634e-19 C

conversions = {
    "m/s": 2.99792458e8,
    "J*s": 6.62607015e-34/(2*np.pi),
    "J/K": 1.380649e-23,
    "J/m/A^2": 1.25663706212e-6,
    "J": 1.602176634e-19,
    "eV": 1,
    "1": 1
}

conversions["s"] = conversions["J*s"] / conversions["J"]
conversions["m"] = conversions["m/s"] * conversions["s"]
conversions["kg"] = conversions["J"] / conversions["m/s"]**2
conversions["g"] = conversions["kg"] / 1000
conversions["A"] = np.sqrt(conversions["J"] / conversions["m"] / conversions["J/m/A^2"])
conversions["C"] = conversions["A"] * conversions["s"]
conversions["V"] = conversions["J"] / conversions["C"]
conversions["N"] = conversions["J"] / conversions["m"]
conversions["W"] = conversions["J"] / conversions["s"]
conversions["T"] = conversions["J"] / conversions["A"] / conversions["m"]**2 
conversions["K"] = conversions["J"] / conversions["J/K"]
conversions["F"] = conversions["C"] / conversions["V"]
conversions["Hz"] = 1/conversions["s"]
conversions["rad"] = 1

prefixes = {
    'T': 1e-12,
    'G': 1e-9,
    'M': 1e-6,
    'k': 1e-3,
    'd': 1e1,
    'c': 1e2,
    'm': 1e3,
    'u': 1e6,
    'μ': 1e6,
    'p': 1e9,
    'f': 1e12
}

def get_conversion(unit):
    if unit in conversions:
        return conversions[unit]
    prefix, unit = unit[0], unit[1:]
    if prefix in prefixes and unit in conversions:
        return prefixes[prefix]*conversions[unit]

def conversion_ratio(unit):
    parts = re.split("([*/])", unit)
    conversion = 1
    previous = "*"
    for part in parts:
        result = 1
        if previous in ["*", "/"]:
            if '^' in part:
                left, right = part.split("^")
                result = get_conversion(left)**int(right)
            else:
                result = get_conversion(part)
            if previous == "/":
                result = 1/result
            conversion *= result
        previous = part
    return conversion

def to(unit, val):
    return val * conversion_ratio(unit)

def fr(unit, val):
    return val / conversion_ratio(unit)