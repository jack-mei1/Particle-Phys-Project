import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.optimize import curve_fit

# data loader
invariant_mass = pd.read_csv("Jpsimumu_Run2011A.csv")["M"]
mass_spike = invariant_mass[(invariant_mass >= 3.0) & (invariant_mass <= 3.25)] # cut off shitty edges bc expected curve is awful
df = pd.DataFrame(mass_spike)

# bins
bins = np.arange(3.05, 3.155, 0.005) # 3.05 to 3.15
width = bins[1] - bins[0]
midpoint = (bins[:-1] + bins[1:]) / 2 # 3.0525, 3.0575, 3.0625 ... 3.1425 3.1475

# models
observed_counts, _ = np.histogram(df["M"], bins=bins)

# expected function
mu = 3.097 # J/psi mass
def model(m, A, B0, B1, stdev): # gpt equation magic (it's a PDF (probability distr. function) for Gaussian/Normal distributions)
    signal = (A / (stdev*np.sqrt(2*np.pi))) * np.exp(-0.5*((m-mu)/stdev)**2)
    bg = B0 + B1 * m
    return (signal + bg) * width

# fitting
guess = [1000, 10, 0, 0.3] # A, B0, B1, stdev
optimal_params, _ = curve_fit(model, midpoint, observed_counts, p0=guess)
A_fit, B0_fit, B1_fit, stdev_fit = optimal_params

expected_counts = model(midpoint, A_fit, B0_fit, B1_fit, stdev_fit)

# chi-squared (GOF for a fitted model)
mask = expected_counts >= 5 # large counts
obs_mask = observed_counts[mask]
exp_mask = expected_counts[mask]
dof = len(obs_mask) - 4 # different for fitted models; dof = (# categories) - (# params) - 1; no -1 because total counts is not fixed

chi_squared, p_val = chisquare(obs_mask, exp_mask)

print(f"Chi-Square: {chi_squared:.2f}")
print(f"DOF: {dof}")
print(f"Chi-Squared per DOF: {chi_squared / dof:.2f}")
print(f"P-Val: {p_val:.3e}")

# plot
plt.figure(figsize=(10, 6))
plt.bar(midpoint, observed_counts, width=width, label="Observed Counts", color="black", alpha=0.7, edgecolor="white")
plt.plot(midpoint, expected_counts, label="Fitted Model", color="blue", linewidth=5)
plt.xlabel("Invariant Mass (GeV)")
plt.ylabel("Counts")
plt.title("Gaussian Fit for J/ψ Mesons")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
