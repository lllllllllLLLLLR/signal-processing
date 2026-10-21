# Signal Processing Labs

Worked labs for the **M2AI Signal Processing** course (M2, Université Paris-Saclay): FIR/IIR filters, image filtering, and a guitar tuner, with real audio data.

## Highlights

- **Lab 3** (`XuanruDai_RanLu_Lab3.ipynb`) & **Lab 4** (`RanLu_XuanruDai_Lab4.ipynb`) — group labs covering digital filtering with FIR/IIR delays.
- **Task 1 — FIR delay** (`task1_fir_delay.ipynb`) — FIR filter design & signal delay.
- **Task 2 — IIR delay** (`task2_iir_delay.ipynb`) — IIR filter design & delay compensation.
- **Task 3 — Image filtering** (`task3_image_filtering.ipynb`) — spatial filtering on images.
- **Guitar tuner** (`guitar_tuner_analysis.ipynb`) — frequency analysis of guitar strings, plus `update_real_data.py` to regenerate real recordings.

## Quick start

```bash
pip install numpy scipy matplotlib librosa soundfile jupyter
jupyter notebook
```

## Layout

```
├── XuanruDai_RanLu_Lab3.ipynb
├── RanLu_XuanruDai_Lab4.ipynb
├── task1_fir_delay.ipynb
├── task2_iir_delay.ipynb
├── task3_image_filtering.ipynb
├── guitar_tuner_analysis.ipynb
├── update_real_data.py
└── data/            # real audio samples (wav) & test image
```
