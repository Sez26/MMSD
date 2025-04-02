import numpy as np
from scipy.integrate import solve_ivp

# Define the system of ODEs
def ice_sheet_feedback(t, X, params):
    ice_mass, albedo, ocean_temp = X  # Example state variables

    # Define dynamic coefficients
    melt_rate = params["melt_base"] * (1 + params["temp_sensitivity"] * ocean_temp)
    albedo_feedback = params["albedo_base"] - params["albedo_slope"] * ice_mass

    # System of ODEs
    d_ice_mass_dt = -melt_rate * ice_mass
    d_albedo_dt = -albedo_feedback * ice_mass
    d_ocean_temp_dt = params["warming_rate"] * (1 - albedo)

    return [d_ice_mass_dt, d_albedo_dt, d_ocean_temp_dt]

# Initial conditions and parameters
X0 = [1.0, 0.7, 0.5]  # Initial ice mass, albedo, ocean temperature
params = {
    "melt_base": 0.02,
    "temp_sensitivity": 0.1,
    "albedo_base": 0.6,
    "albedo_slope": 0.05,
    "warming_rate": 0.01,
}

# Solve the system
t_span = (0, 200)  # Time range
sol = solve_ivp(ice_sheet_feedback, t_span, X0, args=(params,), method="RK45", t_eval=np.linspace(0, 200, 1000))

# Plot the results if needed
import matplotlib.pyplot as plt
plt.plot(sol.t, sol.y[0], label="Ice Mass")
plt.plot(sol.t, sol.y[1], label="Albedo")
plt.plot(sol.t, sol.y[2], label="Ocean Temp")
plt.legend()
plt.xlabel("Time")
plt.ylabel("State Variables")
plt.show()
