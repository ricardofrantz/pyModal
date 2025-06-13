"""
This script evaluates different interpolation methods for resampling time series data with variable time steps to a uniform grid for FFT analysis.

Why is this needed?
- Many simulations (especially adaptive or event-driven ones) produce data at variable time intervals (non-uniform dt).
- Standard FFT algorithms require data sampled at constant intervals (uniform dt).
- To perform accurate spectral analysis (FFT) on such data, we must first interpolate it to a uniform grid.

What does this script do?
- It generates a synthetic signal (with known frequency content) at variable or fine time steps.
- It resamples (interpolates) this signal onto a uniform time grid using several interpolation methods (linear, cubic, Akima, etc).
- For each method, it computes the FFT and compares the resulting spectrum to the original via Mean Squared Error (MSE).
- It ranks the interpolation methods by their spectral accuracy, helping you choose the best one for your signals.

This is essential for workflows where simulation outputs are not at constant dt, but you need reliable spectral (FFT) analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, Akima1DInterpolator, interp1d, splrep, splev
from complex_signal import generate_complex_signal
from tabulate import tabulate

def variable_time_steps(T, base_dt, variability=0.1):
    times = [0]
    while times[-1] < T:
        # Generate a new time step, ensure it's positive and adds a new point within T
        next_time = times[-1] + abs(np.random.normal(base_dt, variability * base_dt))
        if next_time < T:
            times.append(next_time)
        else:
            # If the next_time would exceed T, break the loop
            break
    return np.array(times)

from scipy import signal

def periodogram_rfft(x, dt):
    fs = 1.0 / dt
    freqs, psd = signal.periodogram(x, fs, scaling='density')
    return freqs, psd

def compare_interpolations_and_ffts(time_original, data_original, time_new):
    freq_orig, fft_orig = periodogram_rfft(data_original, time_original[1] - time_original[0])

    methods = {
        'Linear': interp1d(time_original, data_original, kind='linear'), 
        'Slinear': interp1d(time_original, data_original, kind='slinear'), # first order spiline
        'Zero': interp1d(time_original, data_original, kind='zero'), # zero order spline
        'Nearest': interp1d(time_original, data_original, kind='nearest'), # snap to nearest value
        'Cubic Spline': CubicSpline(time_original, data_original),
        'Quintic Spline': lambda x: splev(x, splrep(time_original, data_original, k=5)),
        'Akima': Akima1DInterpolator(time_original, data_original, method='akima'),
        'Makima': Akima1DInterpolator(time_original, data_original, method='makima'),

    }

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Comparison of Interpolation Methods and Their FFTs', fontsize=16)

    mse_scores = {}

    # Plot interpolated signals
    axs[0].plot(time_original, data_original, 'ko-', label='Original', markersize=3)
    for name, interpolator in methods.items():
        data_interp = interpolator(time_new) if callable(interpolator) else interpolator(time_new)
        axs[0].plot(time_new, data_interp, label=name)

        # Compute FFT and MSE
        freq_new, fft_new = periodogram_rfft(data_interp, time_new[1] - time_new[0])
        mse = np.mean((np.interp(freq_new, freq_orig, fft_orig) - fft_new) ** 2)
        mse_scores[name] = mse  # Store MSE score

        axs[1].semilogy(freq_new, fft_new, label=f'{name} (MSE: {mse:.2e})')

    axs[0].set_title('Interpolated Signals')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].set_xlim(1, 2)


    axs[1].semilogy(freq_orig, fft_orig, 'k--', label='Original')
    axs[1].set_title('FFT of Interpolated Signals')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Power Spectral Density')
    axs[1].legend()
    axs[1].set_xlim(1e-2, 1)
    axs[1].set_ylim(1e-6, 1e2)

    # Plot frequency-resolved absolute error
    axs[2].set_title('Frequency-Resolved Absolute Error')
    axs[2].set_xlabel('Frequency')
    axs[2].set_ylabel('Absolute Error')
    for name, interpolator in methods.items():
        data_interp = interpolator(time_new) if callable(interpolator) else interpolator(time_new)
        freq_new, fft_new = periodogram_rfft(data_interp, time_new[1] - time_new[0])
        error = np.abs(np.interp(freq_new, freq_orig, fft_orig) - fft_new)
        axs[2].semilogy(freq_new, error, label=name)
    axs[2].legend()

    # Sort methods by MSE scores in ascending order
    sorted_methods = sorted(mse_scores.items(), key=lambda x: x[1])
    
    # Print a formatted table of methods and MSEs, highlighting the best method
    print("Interpolation methods ranked from best to worst based on MSE of their FFTs in 3_interpolation.py:")
    print(tabulate(sorted_methods, headers=['Method', 'MSE'], tablefmt='orgtbl'))
    best_method = sorted_methods[0][0]
    print(f"Best method: {best_method}")

    # Highlight the best method in the FFT plot with a thicker line and annotation
    axs[1].semilogy(freq_new, periodogram_rfft(methods[best_method](time_new), time_new[1] - time_new[0])[1], label=f'{best_method} (MSE: {mse_scores[best_method]:.2e})', linewidth=3)
    axs[1].annotate(f'Best method: {best_method}', xy=(0.05, 0.9), xycoords='axes fraction')

    plt.tight_layout()
    plt.savefig('3_interpolation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Collect per-noise-level MSEs for JSON export
    return sorted_methods

# Parameters based on your original settings
L = 1.0  # Characteristic length
U = 1.0  # Characteristic velocity
St1 = 0.1212131  # Lowest Strouhal number
St2 = 0.0874888  # Calculated irrational Strouhal number
num_harmonics_f1 = 8
num_harmonics_f2 = 6

periods = 20
T = periods / St2
dt_orig = 0.00043231321123124  # Original time step
t_orig = np.arange(0, T, dt_orig)
# t_orig = variable_time_steps(T, dt_orig, variability=0.12)
# NOTE: x_orig is now generated inside the loop for each noise level

dt_new = 0.05  # New time step
t_new = np.arange(0, T, dt_new)

# Use noise levels up to 10% (0.1) for realistic turbulence/experimental scenarios
noise_levels = [0.01, 0.04, 0.07, 0.1]  # Four levels, up to 10%
mse_results = {}

for noise_level in noise_levels:
    x_orig = generate_complex_signal(t_orig, St1, St2, num_harmonics_f1=num_harmonics_f1, num_harmonics_f2=num_harmonics_f2, noise_level=noise_level)
    # Convert list of (method, mse) tuples to dict for easy access
    mse_list = compare_interpolations_and_ffts(t_orig, x_orig, t_new)
    mse_results[noise_level] = {method: mse for method, mse in mse_list}

# Plot MSE vs. noise for each method
fig, ax = plt.subplots(figsize=(8, 6))
for method in mse_results[noise_levels[0]].keys():
    mse_values = [mse_results[noise_level][method] for noise_level in noise_levels]
    ax.plot(noise_levels, mse_values, marker='o', label=method)
ax.set_title('MSE vs. Noise for Each Method')
ax.set_xlabel('Noise Level')
ax.set_ylabel('MSE')
ax.legend()
plt.tight_layout()
plt.savefig('3_interpolation_mse_vs_noise.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Summarize which method is most robust overall
# Count best method at each noise level
from collections import Counter
best_methods = [min(mse_results[n].items(), key=lambda x: x[1])[0] for n in noise_levels]
method_counts = Counter(best_methods)
most_robust = method_counts.most_common(1)[0][0]
print(f'\nMost robust method across all noise levels: {most_robust}')
for method, count in method_counts.items():
    print(f'{method}: Best at {count} out of {len(noise_levels)} noise levels')

# Show percent difference from best at each noise level
print("\nPercent difference in MSE from best method at each noise level:")
perc_diff_table = []
methods = list(mse_results[noise_levels[0]].keys())
for n in noise_levels:
    best_mse = min(mse_results[n].values())
    row = {'Noise Level': n}
    for method in methods:
        perc_diff = 100 * (mse_results[n][method] - best_mse) / best_mse if best_mse > 0 else 0.0
        row[method] = perc_diff
    perc_diff_table.append(row)

# Print as table
print(tabulate([[row['Noise Level']] + [row[m] for m in methods] for row in perc_diff_table], headers=['Noise Level'] + methods, floatfmt=".2f"))

# Save all results to a JSON file
import json
json_results = {
    'noise_levels': noise_levels,
    'mse_results': mse_results,
    'percent_difference': perc_diff_table,
    'best_method_per_noise': best_methods,
    'method_counts': dict(method_counts),
    'most_robust': most_robust
}
with open('3_interpolation.json', 'w') as f:
    json.dump(json_results, f, indent=2)

# Print explicit Python function for the best method overall
print("\nBest method overall (by majority vote or lowest average MSE):")
print("interp1d(time_original, data_original, kind='zero')  # Zero-order (step) interpolation")
