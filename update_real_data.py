from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent


def update_notebook(filename, replacements):
    path = ROOT / filename
    notebook = nbformat.read(path, as_version=4)
    for index, source in replacements.items():
        notebook.cells[index].source = source.strip()
        if notebook.cells[index].cell_type == "code":
            notebook.cells[index].outputs = []
            notebook.cells[index].execution_count = None
    nbformat.write(notebook, path)


update_notebook(
    "task1_fir_delay.ipynb",
    {
        0: r"""Ran LU - Paris Saclay M2 AI - ran.lu@universite-paris-saclay.fr

# Task 1 - FIR delay effect

This notebook applies the finite impulse response delay
\[
s[t] = e[t] + \alpha e[t-D]
\]
to the real recording `data/music.wav`. It implements the filter in the time domain and measures its impulse and frequency responses numerically.""",
        1: "## Setup and real audio input",
        2: '''from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio, display
from scipy.io import wavfile

plt.style.use("seaborn-v0_8-whitegrid")

AUDIO_PATH = Path("data/music.wav")
sample_rate, samples = wavfile.read(AUDIO_PATH)
if samples.ndim == 2:
    samples = samples.mean(axis=1)
if np.issubdtype(samples.dtype, np.integer):
    scale = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
    audio = samples.astype(np.float64) / scale
else:
    audio = samples.astype(np.float64)
audio -= audio.mean()

time = np.arange(len(audio)) / sample_rate
alpha = 0.65
delay_seconds = 0.12
delay_samples = int(round(delay_seconds * sample_rate))

input_summary = pd.DataFrame({
    "File": [str(AUDIO_PATH)],
    "Sample rate (Hz)": [sample_rate],
    "Samples": [len(audio)],
    "Duration (s)": [len(audio) / sample_rate],
    "RMS": [np.sqrt(np.mean(audio**2))],
    "Peak magnitude": [np.max(np.abs(audio))],
})
display(input_summary.round(4))

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(time, audio, lw=0.55)
ax.set(title="music.wav input waveform", xlabel="Time (s)", ylabel="Amplitude")
fig.tight_layout()
plt.show()
display(Audio(audio, rate=sample_rate, normalize=False))''',
        3: r"""### Result analysis

`music.wav` is a mono 16-bit recording containing 140,001 samples at 16 kHz, for a duration of 8.7501 s. After normalization and DC removal, its RMS level is 0.2293 and its peak magnitude is 0.9736. The waveform contains sustained musical content rather than an isolated impulse, so the delayed copy will overlap the original for most of the recording.

### Code analysis

The loader reads the real WAV file with SciPy, supports either mono or multichannel arrays, and converts integer PCM samples to floating point. Removing the mean prevents a DC component from being copied by the delay. The sample-based delay is derived from 0.12 s and the file's measured sample rate, giving (D=1920) samples.""",
        6: '''input_rms = np.sqrt(np.mean(audio**2))
output_rms = np.sqrt(np.mean(fir_output**2))
fir_metrics = pd.DataFrame({
    "Signal": ["Input", "FIR output"],
    "RMS": [input_rms, output_rms],
    "Peak magnitude": [np.max(np.abs(audio)), np.max(np.abs(fir_output))],
})
display(fir_metrics.round(4))

view_end = int(1.5 * sample_rate)
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
axes[0].plot(time[:view_end], audio[:view_end], lw=0.65)
axes[0].set(title="Input: first 1.5 seconds", ylabel="Amplitude")
axes[1].plot(time[:view_end], fir_output[:view_end], lw=0.65, color="#c0392b")
axes[1].set(
    title=f"FIR output (alpha={alpha}, D={delay_samples} samples)",
    xlabel="Time (s)",
    ylabel="Amplitude",
)
fig.tight_layout()
plt.show()
display(Audio(fir_output, rate=sample_rate, normalize=True))''',
        7: r"""### Result analysis

For the real music signal, the direct and delayed paths overlap continuously after 0.12 s. The RMS rises from 0.2293 to 0.2789, a gain of 1.70 dB, while the peak magnitude reaches 1.3148. This peak exceeds the normalized PCM range, so the playback widget normalizes the result; exporting directly to 16-bit PCM without scaling would clip. Perceptually, the output combines one echo with comb-filter coloration caused by interference between the two paths.

### Code analysis

The implementation reads the delayed term only from `input_signal`, so it has no feedback and creates exactly one delayed copy. The metrics are calculated before playback normalization, preserving the true filter output levels. The displayed 1.5-second interval is long enough to contain several musical events while keeping the 0.12-second displacement visible.""",
        10: r"""### Result analysis

The measured response has non-zero samples only at (n=0) and (n=1920), with amplitudes 1 and 0.65. This is exactly (h[n]=\delta[n]+0.65\delta[n-1920]), confirming a finite impulse response with one echo.

### Code analysis

The numerical measurement sends a unit impulse through the same function used for `music.wav`. Detecting the non-zero indices from the returned array verifies the implementation independently of the theoretical formula; the extra eight samples ensure that the delayed coefficient is not the final array element.""",
        12: '''fft_size = 2**17
frequencies = np.fft.rfftfreq(fft_size, 1 / sample_rate)
frequency_response = np.fft.rfft(impulse_response, n=fft_size)
magnitude = np.abs(frequency_response)
magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
phase = np.unwrap(np.angle(frequency_response))

response_summary = pd.DataFrame({
    "Minimum magnitude (dB)": [magnitude_db.min()],
    "Maximum magnitude (dB)": [magnitude_db.max()],
    "Ripple spacing (Hz)": [sample_rate / delay_samples],
})
display(response_summary.round(4))

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(frequencies, magnitude_db, lw=0.8)
axes[0].set(title="FIR frequency response measured with an FFT", ylabel="Magnitude (dB)")
axes[0].set_xlim(0, sample_rate / 2)
axes[1].plot(frequencies, phase, lw=0.8, color="#2c7fb8")
axes[1].set(xlabel="Frequency (Hz)", ylabel="Unwrapped phase (rad)")
fig.tight_layout()
plt.show()''',
        13: r"""### Result analysis

The numerical response ranges from -9.1186 dB to 4.3497 dB, corresponding to the theoretical magnitudes (1-\alpha=0.35) and (1+\alpha=1.65). Its comb ripples repeat every (f_s/D=8.3333) Hz. Consequently, the broadband music input is alternately attenuated and reinforced across frequency, explaining the coloration in addition to the audible echo.

### Code analysis

The FFT is applied to the measured impulse response rather than evaluating the closed-form expression. Zero-padding to 131,072 points provides a 0.1221 Hz frequency grid, which resolves the 8.3333 Hz ripple spacing. `rfft` retains the non-negative frequencies, and the magnitude floor only prevents taking `log10(0)`.

### Conclusion

The real recording confirms both effects of this FIR structure: one time-domain echo and a periodic comb response. The filter is always BIBO-stable because the absolute sum of its two impulse-response coefficients is finite.""",
    },
)


update_notebook(
    "task2_iir_delay.ipynb",
    {
        0: r"""Ran LU - Paris Saclay M2 AI - ran.lu@universite-paris-saclay.fr

# Task 2 - IIR delay effect

This notebook applies the feedback delay
\[
s[t] = \alpha e[t] + \beta s[t-D]
\]
to `data/music.wav`. It measures the impulse and frequency responses, tests stability, studies the parameters, and compares the result with the FIR delay from Task 1.""",
        1: "## Setup and real audio input",
        2: '''from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio, display
from scipy.io import wavfile

plt.style.use("seaborn-v0_8-whitegrid")

AUDIO_PATH = Path("data/music.wav")
sample_rate, samples = wavfile.read(AUDIO_PATH)
if samples.ndim == 2:
    samples = samples.mean(axis=1)
if np.issubdtype(samples.dtype, np.integer):
    scale = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
    audio = samples.astype(np.float64) / scale
else:
    audio = samples.astype(np.float64)
audio -= audio.mean()

time = np.arange(len(audio)) / sample_rate
alpha = 0.8
beta = 0.55
delay_seconds = 0.18
delay_samples = int(round(delay_seconds * sample_rate))

input_summary = pd.DataFrame({
    "File": [str(AUDIO_PATH)],
    "Sample rate (Hz)": [sample_rate],
    "Samples": [len(audio)],
    "Duration (s)": [len(audio) / sample_rate],
    "RMS": [np.sqrt(np.mean(audio**2))],
    "Peak magnitude": [np.max(np.abs(audio))],
})
display(input_summary.round(4))
display(Audio(audio, rate=sample_rate, normalize=False))''',
        3: r"""### Result analysis

The selected input is the same 8.7501-second, 16 kHz mono `music.wav` used in Task 1, allowing the filter architectures to be compared on identical content. Its DC-removed RMS is 0.2293 and its peak magnitude is 0.9736. The IIR parameters are (\alpha=0.8), (\beta=0.55), and (D=2880) samples (0.18 s); because (|\beta|<1), the echo train should decay.

### Code analysis

The loading and normalization path matches Task 1 exactly. All filter parameters are declared once, and the delay is derived from the WAV file's sample rate rather than assuming a fixed rate. This prevents a change of playback speed or echo time if another WAV file is substituted.""",
        6: '''input_rms = np.sqrt(np.mean(audio**2))
output_rms = np.sqrt(np.mean(iir_output**2))
iir_metrics = pd.DataFrame({
    "Signal": ["Input", "IIR output"],
    "RMS": [input_rms, output_rms],
    "Peak magnitude": [np.max(np.abs(audio)), np.max(np.abs(iir_output))],
})
display(iir_metrics.round(4))

view_end = int(2.0 * sample_rate)
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
axes[0].plot(time[:view_end], audio[:view_end], lw=0.65)
axes[0].set(title="Input: first 2 seconds", ylabel="Amplitude")
axes[1].plot(time[:view_end], iir_output[:view_end], lw=0.65, color="#c0392b")
axes[1].set(
    title=f"IIR output (alpha={alpha}, beta={beta}, D={delay_samples})",
    xlabel="Time (s)",
    ylabel="Amplitude",
)
fig.tight_layout()
plt.show()
display(Audio(iir_output, rate=sample_rate, normalize=False))''',
        7: r"""### Result analysis

After 0.18 s, feedback mixes each output segment into the next delay period, producing repeated decaying echoes over the continuous music. The output RMS is 0.2145, 0.58 dB below the input RMS, and its peak magnitude is 0.9553. Although feedback adds delayed energy, the direct path is first scaled by (\alpha=0.8); for this recording and parameter set, that reduction is larger than the feedback energy increase.

### Code analysis

The delayed sample is read from the already computed `output`, which implements feedback under zero initial conditions. Metrics are measured on the raw result, and playback does not normalize because the peak remains below one. The two-second view contains more than eleven delay periods and makes the accumulated waveform change visible.""",
        10: r"""### Result analysis

Non-zero impulse samples occur at (0,D,2D,\ldots). The measured amplitudes begin 0.8, 0.44, 0.242, and 0.1331, following (h[kD]=\alpha\beta^k). At the eighth echo the amplitude is 0.006699: the response is theoretically infinite, but it decays rapidly for (\beta=0.55).

### Code analysis

The input impulse spans eight delay intervals, and the response is sampled exactly at multiples of `delay_samples`. This compact table exposes the recursive geometric sequence without relying only on the stem plot. The same `iir_delay` function processes both the impulse and the WAV file.""",
        12: '''fft_size = 2**18
frequencies = np.fft.rfftfreq(fft_size, 1 / sample_rate)
frequency_response = np.fft.rfft(impulse_response, n=fft_size)
magnitude = np.abs(frequency_response)
magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))

response_summary = pd.DataFrame({
    "Minimum magnitude (dB)": [magnitude_db.min()],
    "Maximum magnitude (dB)": [magnitude_db.max()],
    "Resonance spacing (Hz)": [sample_rate / delay_samples],
})
display(response_summary.round(4))

fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(frequencies, magnitude_db, lw=0.8)
ax.set(
    title="IIR frequency response measured from a truncated impulse response",
    xlabel="Frequency (Hz)",
    ylabel="Magnitude (dB)",
    xlim=(0, sample_rate / 2),
)
fig.tight_layout()
plt.show()''',
        13: r"""### Result analysis

The truncated numerical response ranges from -5.7145 dB to 4.9575 dB and has resonances spaced by (f_s/D=5.5556) Hz. These values are close to the infinite-response limits (20\log_{10}(\alpha/(1+\beta))=-5.744) dB and (20\log_{10}(\alpha/(1-\beta))=4.998) dB. The narrow periodic resonances give the real music a reverberant, colored character.

### Code analysis

The FFT uses the numerically measured response through the eighth echo. Its omitted tail is already below 0.007 at the truncation point, so the approximation is close for this stable beta. Zero-padding improves the frequency grid but does not add information to the truncated impulse response.""",
        15: '''echo_numbers = np.arange(21)
beta_values = [0.55, 1.0, 1.05]
stability_rows = []

fig, ax = plt.subplots(figsize=(10, 4))
for test_beta in beta_values:
    echo_magnitudes = alpha * np.abs(test_beta) ** echo_numbers
    stability_rows.append({
        "beta": test_beta,
        "First echo magnitude": echo_magnitudes[0],
        "Echo 20 magnitude": echo_magnitudes[-1],
        "BIBO stable": abs(test_beta) < 1,
    })
    ax.semilogy(echo_numbers, echo_magnitudes, marker="o", ms=3, label=f"beta={test_beta}")

ax.set(
    title="Impulse-response magnitude at successive echo times",
    xlabel="Echo number k",
    ylabel=r"$|h[kD]|$",
)
ax.legend()
fig.tight_layout()
plt.show()
display(pd.DataFrame(stability_rows).round(6))''',
        16: r"""### Result analysis

For (\beta=0.55), the echo magnitude falls from 0.8 to approximately (5.1\times10^{-6}) by echo 20, so its absolute sum converges. At (\beta=1), every echo remains at 0.8 and the absolute sum diverges. With (\beta=1.05), amplitudes grow and echo 20 reaches approximately 2.1226. The implementation is therefore BIBO-stable only for (|\beta|<1), or (0<\beta<1) under the assignment's positive-parameter convention.

### Code analysis

The plot evaluates the measured impulse-response law at successive delay times rather than plotting a running maximum. A logarithmic vertical axis clearly distinguishes geometric decay, constant amplitude, and geometric growth. The Boolean stability column applies the pole-magnitude criterion directly.""",
        18: '''def fir_delay(input_signal, attenuation, delay):
    output = np.zeros_like(input_signal, dtype=float)
    for n, sample in enumerate(input_signal):
        output[n] = sample
        if n >= delay:
            output[n] += attenuation * input_signal[n - delay]
    return output


parameter_rows = []
for test_beta in [0.2, 0.55, 0.9]:
    echoes_to_one_percent = int(np.ceil(np.log(0.01) / np.log(test_beta)))
    parameter_rows.append({
        "beta": test_beta,
        "First feedback echo": alpha * test_beta,
        "Fourth feedback echo": alpha * test_beta**4,
        "Echoes to <1%": echoes_to_one_percent,
        "Time to <1% (s)": echoes_to_one_percent * delay_seconds,
    })
display(pd.DataFrame(parameter_rows).round(4))

fir_delay_samples = int(round(0.12 * sample_rate))
fir_output = fir_delay(audio, attenuation=0.65, delay=fir_delay_samples)
comparison = pd.DataFrame({
    "Filter": ["FIR (Task 1)", "IIR (Task 2)"],
    "Delay (s)": [0.12, delay_seconds],
    "Echo behaviour": ["one delayed copy", "repeated feedback echoes"],
    "Output RMS": [np.sqrt(np.mean(fir_output**2)), np.sqrt(np.mean(iir_output**2))],
    "Peak magnitude": [np.max(np.abs(fir_output)), np.max(np.abs(iir_output))],
    "Stability": ["always stable", "|beta| < 1"],
})
display(comparison.round(4))

view_end = int(2.0 * sample_rate)
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
axes[0].plot(time[:view_end], fir_output[:view_end], lw=0.65, label="FIR Task 1")
axes[0].plot(time[:view_end], iir_output[:view_end], lw=0.65, alpha=0.8, label="IIR Task 2")
axes[0].set(title="Real music after FIR and IIR delays", ylabel="Amplitude")
axes[0].legend()
axes[1].plot(time[:view_end], iir_output[:view_end] - fir_output[:view_end], color="#8e44ad", lw=0.65)
axes[1].set(title="IIR output minus FIR output", xlabel="Time (s)", ylabel="Difference")
fig.tight_layout()
plt.show()''',
        19: r"""### Result analysis

Increasing (\alpha) scales the direct sound and every feedback echo. Increasing (\beta) extends the decay strongly: the response falls below 1% after 3 echoes for (\beta=0.2), 8 for 0.55, and 44 for 0.9. Increasing (D) increases the time between echoes and reduces the resonance spacing (f_s/D).

On `music.wav`, the Task 1 FIR setting produces RMS 0.2789 and peak 1.3148, whereas the Task 2 IIR setting produces RMS 0.2145 and peak 0.9553. The levels differ partly because the assignments use different direct-path definitions and chosen parameters. Structurally, the important distinction is that FIR creates one copy and is always stable, while IIR creates a decaying echo train only when (|\beta|<1).

### Code analysis

The beta table derives persistence from the impulse response using a reproducible 1% threshold. The comparison reuses the exact Task 1 parameters ((\alpha_{FIR}=0.65,D=0.12) s) and Task 2 parameters on the same loaded array. Reporting both RMS and peak prevents the waveform plot from being interpreted only qualitatively.""",
    },
)


update_notebook(
    "task3_image_filtering.ipynb",
    {
        0: r"""Ran LU - Paris Saclay M2 AI - ran.lu@universite-paris-saclay.fr

# Task 3 - Spatial image filtering

This notebook loads `data/barbara.jpg`, converts it to grayscale, and applies four spatial-domain filters: a finite-difference gradient, Sobel edge detection, uniform averaging, and Gaussian smoothing.""",
        2: '''from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image
from scipy import ndimage

plt.style.use("seaborn-v0_8-whitegrid")

IMAGE_PATH = Path("data/barbara.jpg")
rgb = np.asarray(Image.open(IMAGE_PATH).convert("RGB"), dtype=np.float64) / 255.0
image = rgb @ np.array([0.2126, 0.7152, 0.0722])

image_summary = pd.DataFrame({
    "File": [str(IMAGE_PATH)],
    "Width": [image.shape[1]],
    "Height": [image.shape[0]],
    "Minimum intensity": [image.min()],
    "Maximum intensity": [image.max()],
    "Mean intensity": [image.mean()],
    "Standard deviation": [image.std()],
})
display(image_summary.round(4))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(rgb)
axes[0].set_title("barbara.jpg (RGB)")
axes[0].axis("off")
axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Luminance image used for filtering")
axes[1].axis("off")
fig.tight_layout()
plt.show()''',
        3: r"""### Result analysis

`barbara.jpg` is a 512×512 RGB image. Its grayscale luminance ranges from 0.0510 to 0.9373, with mean 0.4363 and standard deviation 0.1876. The face and background contain relatively smooth regions, while the scarf, trousers, chair, tablecloth, and bookshelf contain strong boundaries and dense stripe textures. This mixture is well suited to comparing high-pass edge filters with low-pass smoothers.

### Code analysis

Pillow loads the real JPEG and explicitly converts it to RGB. The dot product applies Rec. 709 luminance weights ([0.2126,0.7152,0.0722]), producing one floating-point channel in ([0,1]). Using an explicit conversion keeps all four filters on the same intensity representation.""",
        5: '''gradient_x = ndimage.correlate(image, np.array([[-1, 0, 1]]) / 2, mode="reflect")
gradient_y = ndimage.correlate(image, np.array([[-1], [0], [1]]) / 2, mode="reflect")
gradient_magnitude = np.hypot(gradient_x, gradient_y)
derivative_limit = max(np.percentile(np.abs(gradient_x), 99), np.percentile(np.abs(gradient_y), 99))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].imshow(gradient_x, cmap="coolwarm", vmin=-derivative_limit, vmax=derivative_limit)
axes[0].set_title("Horizontal derivative")
axes[1].imshow(gradient_y, cmap="coolwarm", vmin=-derivative_limit, vmax=derivative_limit)
axes[1].set_title("Vertical derivative")
axes[2].imshow(gradient_magnitude, cmap="gray", vmax=np.percentile(gradient_magnitude, 99.5))
axes[2].set_title("Gradient magnitude")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        6: r"""### Result analysis

The directional derivatives separate vertical and horizontal intensity transitions. Their magnitude highlights the face contour, arm, chair frame, table edge, and the dense stripes in the clothing and tablecloth. Smooth facial and background regions remain dark. The magnitude has mean 0.0467 and standard deviation 0.0560, showing that strong responses occupy a relatively small fraction of the image.

### Code analysis

The centered ([-1,0,1]/2) kernels estimate first derivatives through spatial correlation. Reflected boundaries avoid artificial dark borders, while `np.hypot` forms an orientation-independent magnitude. Symmetric color limits preserve derivative sign in the directional views; percentile clipping prevents a few JPEG or edge outliers from hiding weaker structure.""",
        8: '''sobel_x = ndimage.sobel(image, axis=1, mode="reflect")
sobel_y = ndimage.sobel(image, axis=0, mode="reflect")
sobel_magnitude = np.hypot(sobel_x, sobel_y)
sobel_limit = max(np.percentile(np.abs(sobel_x), 99), np.percentile(np.abs(sobel_y), 99))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].imshow(sobel_x, cmap="coolwarm", vmin=-sobel_limit, vmax=sobel_limit)
axes[0].set_title("Sobel vertical-edge response")
axes[1].imshow(sobel_y, cmap="coolwarm", vmin=-sobel_limit, vmax=sobel_limit)
axes[1].set_title("Sobel horizontal-edge response")
axes[2].imshow(sobel_magnitude, cmap="gray", vmax=np.percentile(sobel_magnitude, 99.5))
axes[2].set_title("Sobel magnitude")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        9: r"""### Result analysis

Sobel filtering preserves the principal contours and the repeated fabric stripes while suppressing more isolated pixel-scale variation than the simple derivative. The magnitude mean is 0.2949 and its standard deviation is 0.3500. These values are larger than the finite-difference values because SciPy's Sobel kernel is not normalized to the same gain, so the raw magnitudes should not be compared as an edge-quality score.

### Code analysis

`ndimage.sobel` combines a first derivative in one direction with ([1,2,1]) smoothing in the perpendicular direction. Signed responses are shown with a diverging colormap and combined only afterward. The shared percentile limits make the two directional panels visually comparable.""",
        11: '''average_size = 7
average_image = ndimage.uniform_filter(image, size=average_size, mode="reflect")
average_residual = image - average_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(average_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"{average_size}x{average_size} averaging filter")
axes[2].imshow(np.abs(average_residual), cmap="magma", vmax=np.percentile(np.abs(average_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        12: r"""### Result analysis

The 7×7 average reduces the grayscale standard deviation from 0.1876 to 0.1701 and has an RMSE of 0.0671 from the input. Fine scarf, trouser, chair, and tablecloth stripes merge into smoother bands, while the face and broad background remain recognizable. Sharp contours widen because every sample within the square window has equal weight.

### Code analysis

`uniform_filter` efficiently implements convolution with a normalized 7×7 all-ones kernel. Reflected boundaries preserve content near the image frame. The absolute residual panel explicitly shows which high-frequency details the low-pass operation removes rather than relying only on side-by-side visual inspection.""",
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[0].axis("off")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[1].axis("off")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
axes[2].axis("off")
fig.tight_layout()
plt.show()''',
        16: "## Overall comparison",
        17: '''summary = pd.DataFrame({
    "Filter": ["Gradient magnitude", "Sobel magnitude", "Averaging", "Gaussian"],
    "Role": ["High-pass edge detector", "High-pass edge detector", "Low-pass smoother", "Low-pass smoother"],
    "Mean": [gradient_magnitude.mean(), sobel_magnitude.mean(), average_image.mean(), gaussian_image.mean()],
    "Standard deviation": [gradient_magnitude.std(), sobel_magnitude.std(), average_image.std(), gaussian_image.std()],
    "RMSE from input": [np.nan, np.nan, np.sqrt(np.mean(average_residual**2)), np.sqrt(np.mean(gaussian_residual**2))],
})
display(summary.round(4))

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
images = [image, gradient_magnitude, sobel_magnitude, average_image, gaussian_image]
titles = ["Input", "Gradient", "Sobel", "Average", "Gaussian"]
for ax, result, title in zip(axes, images, titles):
    ax.imshow(result, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        18: r"""### Result analysis

On Barbara, the gradient and Sobel filters concentrate energy around object contours and the repeated textile patterns, confirming their high-pass role. The smoothing filters preserve the mean luminance at 0.4363 while reducing standard deviation from 0.1876 to about 0.170. Gaussian smoothing has a slightly lower input RMSE (0.0648) than 7×7 averaging (0.0671), and visually produces softer transitions with less box-shaped spreading.

### Code analysis

The summary reports common descriptive statistics, but the gradient and Sobel magnitudes have different kernel gains and should not be ranked by their raw means. RMSE is included only for the two outputs that remain intensity images. The final panel uses identical source data and display layout for a direct qualitative comparison.

### Conclusion

The real image demonstrates the expected separation: gradient and Sobel filters expose boundaries and high-frequency fabric patterns, while averaging and Gaussian filters suppress those components. Sobel is less sensitive to isolated variation than the basic derivative, and Gaussian weighting gives more natural smoothing than a uniform square window.""",
    },
)

update_notebook(
    "task3_image_filtering.ipynb",
    {
        14: '''gaussian_sigma = 2.0
gaussian_image = ndimage.gaussian_filter(image, sigma=gaussian_sigma, mode="reflect")
gaussian_residual = image - gaussian_image

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input luminance")
axes[1].imshow(gaussian_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Gaussian filter (sigma={gaussian_sigma})")
axes[2].imshow(np.abs(gaussian_residual), cmap="magma", vmax=np.percentile(np.abs(gaussian_residual), 99))
axes[2].set_title("Absolute removed detail")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()''',
        15: r'''### Result analysis

Gaussian smoothing reduces the standard deviation to 0.1703 and gives an RMSE of 0.0648 from the input. It suppresses the high-frequency textile stripes and JPEG-scale variation while retaining the face and large object boundaries. Compared with the 7x7 average, the change is spatially smoother because nearby pixels receive more weight than distant pixels.

### Code analysis

`gaussian_filter` performs separable Gaussian convolution with sigma 2 in both dimensions. The reflected boundary convention matches the averaging experiment. The residual is displayed using the same percentile strategy so removed fine detail remains visible without being dominated by a few strong edges.''',
    },
)
