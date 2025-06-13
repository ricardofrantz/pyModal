import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq
import time
from complex_signal import generate_complex_signal

t0 = time.perf_counter()

def periodogram_rfft(x, fs):
    """
    Compute the power spectral density (PSD) of a real-valued signal using the periodogram method and the real FFT (RFFT).

    This function computes:
        psd = |RFFT(x)|^2 / (N * fs)
    where N = len(x), fs is the sampling frequency, and RFFT(x) is the one-sided FFT of x.

    - The output `freqs` are the non-negative frequency bins (Hz), compatible with the RFFT.
    - The output `psd` is in units of signal_power_per_Hz (e.g., V^2/Hz for voltage signals).
    - This is the standard periodogram definition for signals sampled at `fs` Hz, and matches the output of scipy.signal.periodogram with scaling='density'.
    - The RFFT is used for efficiency: for real-valued input, only the non-negative frequency bins are needed and computed.

    NOTE ON NORMALIZATION:
    The normalization here (psd = |X|^2 / (N * fs)) is correct for standard unnormalized FFT conventions (as used by numpy, scipy, etc).
    Do NOT apply any additional normalization to X or the PSD result. Double normalization (e.g., dividing by N or fs again)
    will lead to incorrect spectral amplitudes and energies. This is a common source of error when porting code or switching FFT libraries.
    """
    # N = len(x)
    # X = rfft(x)
    # freqs = rfftfreq(N, 1/fs)
    # psd = np.abs(X)**2 / (N * fs)

    freqs, psd = signal.periodogram(x, fs, scaling='density')

    # This manual computation is equivalent to:
    #     scipy.signal.periodogram(x, fs, scaling='density')
    #
    # 'scaling="density"' gives the Power Spectral Density (PSD) in physical units (e.g., V^2/Hz),
    # matching the normalization used here (|X|^2 / (N * fs)).
    #
    # If you use 'scaling="spectrum"', the periodogram instead returns the power per frequency bin (|X|^2 / N^2),
    # which is NOT normalized by the frequency resolution (Hz). This is sometimes called the "power spectrum" rather than PSD.
    #
    # When to use each:
    #   - Use 'density' (the default here and in most scientific work) when you want power per Hz (PSD),
    #     which is appropriate for comparing signals of different lengths or sampling rates, or for physical interpretation.
    #   - Use 'spectrum' if you want the raw squared amplitude per FFT bin, e.g., for mathematical analysis or when integrating over bins.
    #

    return freqs, psd

def blackman_tukey_rfft(x, fs):
    """
    Compute power spectral density (PSD) using the Blackman-Tukey method with the real FFT (RFFT).
    Frequencies are normalized (f = St).

    - The autocorrelation of the signal is computed and windowed (here, no explicit window is applied).
    - The PSD is obtained by taking the RFFT of the autocorrelation and normalizing by the sampling frequency fs.
    - Output `freqs` are the non-negative frequency bins (Hz), compatible with the RFFT.
    - Output `psd` is in units of signal_power_per_Hz (e.g., V^2/Hz for voltage signals), matching the standard Blackman-Tukey convention.

    NOTE ON NORMALIZATION:
    The normalization here (psd = |RFFT(autocorr)| / fs) is standard for the Blackman-Tukey method with unnormalized FFTs.
    Do NOT apply any additional normalization to the FFT output or PSD. Double normalization will give incorrect results.
    This is a common source of error when porting code or switching FFT libraries.

    EQUIVALENT IN SCIPY:
    There is no direct one-liner equivalent for the Blackman-Tukey method in scipy.signal.
    Related functions are:
      - scipy.signal.csd: Computes the cross-spectral density (including autocorrelation for PSD), but uses Welch's method by default.
      - scipy.signal.welch: Computes PSD using Welch's (averaged periodogram) method, not Blackman-Tukey.
    To replicate Blackman-Tukey exactly, you must compute the autocorrelation and then its FFT, as done here.
    """
    N = len(x)
    autocorr = signal.correlate(x, x, mode='full') / N
    autocorr = autocorr[N-1:]
    X = rfft(autocorr)
    freqs = rfftfreq(N, 1/fs)
    psd = np.abs(X) / fs

    return freqs, psd

def welch_method(x, fs):
    """
    Compute power spectral density (PSD) using Welch's method (averaged periodogram), via scipy.signal.welch.
    Frequencies are normalized (f = St).

    - Welch's method splits the signal into overlapping segments, computes a periodogram for each, and averages them.
    - This reduces the variance of the PSD estimate at the cost of frequency resolution.
    - Output `freqs` are the non-negative frequency bins (Hz).
    - Output `psd` is in units of signal_power_per_Hz (e.g., V^2/Hz for voltage signals), matching the standard PSD convention.
    - The parameter scaling='density' gives the PSD (power per Hz), which is the default and standard for physical interpretation.
    - If you use scaling='spectrum', the output is the power per FFT bin (not per Hz).
    - Always check the scaling and normalization to ensure correct interpretation and comparison across libraries and publications.

    This function is a direct wrapper for scipy.signal.welch with scaling='density'.
    """
    freqs, psd = signal.welch(x, fs, nperseg=len(x), scaling='density')
    return freqs, psd


# Find and print peak Strouhal numbers
def find_peaks(st, psd, threshold=0.01):
    peak_indices = signal.find_peaks(psd, height=max(psd)*threshold)[0]
    return st[peak_indices], psd[peak_indices]

# Function to calculate error
def calculate_error(detected_peaks, true_peaks):
    errors = []
    for true_peak in true_peaks:
        if len(detected_peaks) > 0:
            error = min(abs(detected_peak - true_peak) for detected_peak in detected_peaks)
            errors.append(error)
        else:
            errors.append(true_peak)  # If no peaks detected, consider the true peak as the error
    return np.mean(errors)

# Parameters
St1 = 0.1212131  # Primary normalized frequency (Strouhal number)
St2 = 0.0874888  # Secondary normalized frequency (Strouhal number)

T1 = 1 / St1
T2 = 1 / St2
periods = 10.231312
T = periods * T2
fs = 1000
N = int(T * fs)

# Generate the signal
t = np.arange(0, T, 1/fs)
x = generate_complex_signal(t, St1, St2)

# True peaks (St1, St2, and their harmonics)
true_peaks = [St2] + [i*St1 for i in range(1, 6)] + [i*St2 for i in range(2, 4)]

results = {}

methods = [
    ("Periodogram", periodogram_rfft),
    ("Blackman-Tukey", blackman_tukey_rfft),
    ("Welch", welch_method)
]

for method_name, method_func in methods:
    start_time = time.perf_counter()
    st, psd = method_func(x, fs)
    execution_time = time.perf_counter() - start_time
    peaks_st, _ = find_peaks(st, psd)
    error = calculate_error(peaks_st, true_peaks)
    results[method_name] = {
        "st": st,
        "psd": psd,
        "peaks": peaks_st,
        "time": execution_time,
        "error": error
    }

# Print results
print("Method Results:")
for method, data in results.items():
    print(f"{method}:")
    print(f"  Execution time: {data['time']:.4f} seconds")
    print(f"  Error: {data['error']:.6f}")
    print(f"  Peaks: {', '.join([f'{peak:.4f}' for peak in data['peaks']])}")
    print()

# Print frequency resolution (normalized)
st_resolution = fs / N
print(f"Frequency (St) resolution: {st_resolution:.6f}")

# Plotting
size_factor = 0.6
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8*size_factor, 8*size_factor), height_ratios=[1, 3])

# Plot the original signal
ax1.plot(t, x)
ax1.set_title('Signal with Harmonics and Modulation')
ax1.set_xlabel('t')
ax1.set_ylabel('Amp.')

# Define custom start and end times for the inset
t_start = 1*T2  # Example start time
t_end = 2*T2    # Example end time

# Create inset axes
ax1_inset = ax1.inset_axes([0.65, 0.25, 0.33, 0.4])

# Find indices corresponding to t_start and t_end
i_start = np.searchsorted(t, t_start)
i_end = np.searchsorted(t, t_end)

# Plot zoomed-in version of the signal
ax1_inset.plot(t[i_start:i_end], x[i_start:i_end])
ax1_inset.set_xlabel('t', fontsize=8)
ax1_inset.set_ylabel('Amp.', fontsize=8)
ax1_inset.tick_params(axis='both', which='major', labelsize=8)

# Add rectangle patch to main axes to show zoomed area
rect = plt.Rectangle((t_start, min(x[i_start:i_end])), 
                     t_end - t_start,
                     max(x[i_start:i_end]) - min(x[i_start:i_end]),
                     fill=False, color='red', linestyle='--', linewidth=1)
ax1.add_patch(rect)

# Draw lines connecting the rectangle to the inset axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax1, ax1_inset, loc1=2, loc2=4, fc="none", ec="red", linestyle='--')

# Set the limits of the inset axes explicitly
ax1_inset.set_xlim(t_start, t_end)

# Plot the spectra on log-log scale
for method, data in results.items():
    ax2.loglog(data['st'], data['psd'], label=f"{method}")

ax2.set_title('Power Spectral Density Estimates')
ax2.set_xlabel('Strouhal Number (St)')
ax2.set_ylabel('PSD')
ax2.legend()
ax2.set_xlim(0.8*St2, 2)
ax2.set_ylim(1e-4, 1e4)
ax2.grid(True, which="both", ls="-", alpha=0.5)

# Add vertical lines for St1, its harmonics, and St2
for i in range(1, 6):
    ax2.axvline(x=i*St1, color='r', linestyle='--', alpha=0.3)
    ax2.text(i*St1, ax2.get_ylim()[1], f'${i}St_1$', rotation=90, va='top', ha='right', color='r', alpha=0.5)
for i in range(1, 4):
    ax2.axvline(x=i*St2, color='g', linestyle='--', alpha=0.3)
    ax2.text(i*St2, ax2.get_ylim()[1], f'${i}St_2$', rotation=90, va='top', ha='right', color='g', alpha=0.5)

# Adjust layout and save figure
plt.tight_layout()
plt.savefig('4_methods_spectral_analysis.png', dpi=400, bbox_inches='tight')

# Process results
method_performance = {}
for method, data in results.items():
    error_percentage = data['error'] * 100  # Convert to percentage
    time_taken = data['time']
    performance = 1 / error_percentage  # Inverse of error percentage as performance metric
    method_performance[method] = {
        'error_percentage': error_percentage,
        'time': time_taken,
        'performance': performance
    }

# Find method with smallest error
best_method = min(method_performance, key=lambda x: method_performance[x]['error_percentage'])

print(f"Method with smallest error: {best_method}")
print(f"Error: {method_performance[best_method]['error_percentage']:.2f}%")
print(f"Time: {method_performance[best_method]['time']:.4f} seconds")

# Plot performance vs time
plt.figure(figsize=(10, 6))
for method, data in method_performance.items():
    plt.scatter(data['time'], data['performance'], label=method, s=100)

plt.xlabel('Execution Time (seconds)')
plt.ylabel('Performance (1 / Error Percentage)')
plt.title('Performance vs Execution Time for Spectral Methods')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.5)

# Add method names as annotations
for method, data in method_performance.items():
    plt.annotate(method, (data['time'], data['performance']), 
                 xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('4_methods_perf.png', dpi=400, bbox_inches='tight')

# Print performance table
print("\nPerformance Table:")
print("{:<15} {:<20} {:<20} {:<20}".format("Method", "Error (%)", "Time (s)", "Performance"))
print("-" * 75)
for method, data in method_performance.items():
    print("{:<15} {:<20.2f} {:<20.4f} {:<20.4f}".format(
        method, 
        data['error_percentage'], 
        data['time'], 
        data['performance']
    ))

# Noise levels to test
noise_levels = [0.05, 0.1, 0.15, 0.2]

# Results dictionary
results_by_noise = {}

for noise_level in noise_levels:
    periods = 1  # Start with 1 period
    max_periods = 200  # Set a maximum to prevent infinite loops
    
    while periods <= max_periods:
        T = periods * T2  # Total time
        fs = 1000  # Sampling frequency
        N = int(T * fs)  # Number of samples

        # Generate the signal
        t = np.arange(0, T, 1/fs)
        x = generate_complex_signal(t, St1, St2, noise_level=noise_level)

        # True peaks (St1, St2, and their harmonics)
        true_peaks = [St2] + [i*St1 for i in range(1, 6)] + [i*St2 for i in range(2, 4)]

        results = {}

        methods = [
            ("Periodogram", periodogram_rfft),
            ("Blackman-Tukey", blackman_tukey_rfft),
            ("Welch", welch_method),
        ]

        for method_name, method_func in methods:
            st, psd = method_func(x, fs)
            peaks_st, _ = find_peaks(st, psd)
            error = calculate_error(peaks_st, true_peaks)
            results[method_name] = {"error": error}

        # Check if all methods have achieved error < 1%
        if all(data["error"] < 0.02 for data in results.values()):
            break

        periods += 1

    results_by_noise[noise_level] = {"periods": periods, "results": results}

# Print and plot results
plt.figure(figsize=(12, 8))
for noise_level, data in results_by_noise.items():
    periods = data["periods"]
    errors = [method_data["error"] * 100 for method_data in data["results"].values()]
    plt.scatter([noise_level] * len(errors), errors, label=f"{periods} periods")

plt.xlabel("Noise Level")
plt.ylabel("Error Percentage")
plt.title("Error vs Noise Level for Different Numbers of Periods")
plt.legend(title="Periods of T2")
plt.yscale('log')
plt.grid(True)
plt.savefig('4_methods_error_vs_noise.png', dpi=400, bbox_inches='tight')

print("Results:")
for noise_level, data in results_by_noise.items():
    print(f"\nNoise Level: {noise_level}")
    print(f"Periods of T2 needed: {data['periods']}")
    for method, method_data in data['results'].items():
        print(f"  {method}: Error = {method_data['error']*100:.2f}%")

print(f"\nTotal execution time: {time.perf_counter() - t0:.2f} seconds")