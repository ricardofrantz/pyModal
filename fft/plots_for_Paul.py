import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend for file saving
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import rfft, rfftfreq
import time, os

t0 = time.perf_counter()

data_filename = 'fwh_fine_pressure.csv'
root_filename = os.path.splitext(os.path.basename(data_filename))[0]

# Load data from CSV file (expects columns: time,pressure)
csv_data = np.genfromtxt(data_filename, delimiter=',', names=True)
time_arr = csv_data['time']
pressure_arr = csv_data['pressure']

# --- Check for duplicate time values ---
def check_for_duplicates(arr, name="array"):
    unique, counts = np.unique(arr, return_counts=True)
    num_duplicates = np.sum(counts > 1)
    if num_duplicates == 0:
        print(f"No duplicates found in {name}.")
    else:
        print(f"{num_duplicates} duplicate values found in {name}.")

check_for_duplicates(time_arr, name="time array")

# --- Interpolate to uniform grid for FFT analysis ---
# Create a uniform time grid based on data range and mean dt
uniform_dt = np.mean(np.diff(time_arr))
t_uniform = np.arange(time_arr[0], time_arr[-1], uniform_dt)

from scipy.interpolate import interp1d
interp_func = interp1d(time_arr, pressure_arr, kind='cubic', fill_value='extrapolate')
pressure_uniform = interp_func(t_uniform)
print(f"Interpolating to uniform grid: {len(time_arr)} points -> {len(t_uniform)} points")

# --- Apply a very weak, smooth low-pass filter ---
from scipy.signal import butter, filtfilt

def lowpass_filter(data, fs, cutoff_hz, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Set cutoff very high (weak filtering, e.g., 0.95 of Nyquist)
very_weak_cutoff = 0.9 * (0.5 / uniform_dt)  # 95% of Nyquist
pressure_uniform_filtered = lowpass_filter(pressure_uniform, fs=1/uniform_dt, cutoff_hz=very_weak_cutoff, order=2)
print(f"Filtering signal: {len(pressure_uniform)} points (very weak, cutoff={very_weak_cutoff:.2f} Hz)")

# Use t_uniform and pressure_uniform_filtered for all further analysis
t = t_uniform
x = pressure_uniform_filtered

# Estimate sampling frequency from uniform time array
dt = np.mean(np.diff(t))
fs = 1.0 / dt
N = len(x)

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
    freqs, psd = signal.periodogram(x, fs, scaling='density')
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

def find_peaks(st, psd, threshold=0.09):
    peak_indices = signal.find_peaks(psd, height=max(psd)*threshold)[0]
    return st[peak_indices], psd[peak_indices]

# Use loaded data for analysis
t = time_arr
x = pressure_arr

results = {}
methods = [
    ("Periodogram", periodogram_rfft),
    # ("Blackman-Tukey", blackman_tukey_rfft),
    ("Welch", welch_method)
]
for method_name, method_func in methods:
    st, psd = method_func(x, fs)
    peaks_st, peaks_psd = find_peaks(st, psd)
    results[method_name] = {
        "st": st,
        "psd": psd,
        "peaks": peaks_st,
        "peaks_psd": peaks_psd
    }

# Print results
print("Method Results:")
for method, data in results.items():
    print(f"{method}:")
    print(f"  Peaks (Hz): {', '.join([f'{peak:.4f}' for peak in data['peaks']])}")
    print()

freq_resolution = fs / N
print(f"Frequency resolution: {freq_resolution:.6f} Hz")

# Plotting
size_factor = 0.6
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8*size_factor, 8*size_factor), gridspec_kw={'height_ratios': [1, 3]})

# Plot the original signal
ax1.plot(t, x)
# ax1.set_title('Pressure fluctuations')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Pressure [Pa]')

# Inset: zoom on first 1% of data
t_start = t[0] #  + 0.1
t_end = t[0] + 0.01 * (t[-1] - t[0])
ax1_inset = ax1.inset_axes([0.65, 0.25, 0.33, 0.4])
i_start = np.searchsorted(t, t_start)
i_end = np.searchsorted(t, t_end)
ax1_inset.plot(t[i_start:i_end], x[i_start:i_end])
# ax1_inset.set_xlabel('Time [s]', fontsize=8)
# ax1_inset.set_ylabel('Pressure [Pa]', fontsize=8)
# ax1_inset.tick_params(axis='both', which='major', labelsize=8)
rect = plt.Rectangle((t_start, min(x[i_start:i_end])), 
                     t_end - t_start,
                     max(x[i_start:i_end]) - min(x[i_start:i_end]),
                     fill=False, color='red', linestyle='--', linewidth=1)
ax1.add_patch(rect)
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax1, ax1_inset, loc1=2, loc2=4, fc="none", ec="red", linestyle='--')
ax1_inset.set_xlim(t_start, t_end)

# --- Strouhal (St) and Hz dual x-axis ---
# Placeholder: St = f * ST_PLACEHOLDER (user must set correct scaling)
ST_PLACEHOLDER = 1000  # TODO: Set this to D/U_inf or appropriate scaling for your case

for method, data in results.items():
    ax2.loglog(data['st'], data['psd'], label=f"{method}")
    # Mark peaks
    ax2.scatter(data['peaks'], data['peaks_psd'], marker='x', s=60, label=f"{method} peaks")

ax2.set_xlim(10, 3e4)
ax2.set_ylim(1e-12, 1e-4)
# ax2.set_title('Power Spectral Density')
ax2.set_xlabel('Strouhal Number (St)', labelpad=10)
ax2.set_ylabel('PSD [(unit)^2/Hz]')
ax2.legend()
ax2.grid(True, which="both", ls="-", alpha=0.5)

# # Add upper x-axis for Hz
# from matplotlib.ticker import FuncFormatter

def st_to_hz(st):
    # Inverse of St = f * ST_PLACEHOLDER
    return st / ST_PLACEHOLDER

def hz_to_st(hz):
    return hz * ST_PLACEHOLDER

secax = ax2.secondary_xaxis('top', functions=(st_to_hz, hz_to_st))
secax.set_xlabel('Frequency [Hz]')

plt.tight_layout()
fig_filename = f'{root_filename}.png'
plt.savefig(fig_filename, dpi=500, bbox_inches='tight')
print(f"Saved figure: {fig_filename}")

print(f"\nTotal execution time: {time.perf_counter() - t0:.2f} seconds")