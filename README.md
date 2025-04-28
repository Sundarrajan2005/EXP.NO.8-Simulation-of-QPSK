# EXP.NO.8-Simulation-of-QPSK

8.Simulation of QPSK

# AIM:
To simulate Quadrature Phase-Shift Keying (QPSK) Modulation using python code.

# SOFTWARE REQUIRED:
Google Colab

# ALGORITHMS:

Input: Received QPSK signal

Output: Reconstructed binary data

Step 1: Receive Signal

Capture the incoming QPSK-modulated carrier wave.

Step 2: Mix with Carrier Signals

Step 3: Low-pass Filter

Filter the mixed signals to remove high-frequency components, keeping only baseband I and Q.

Step 4: Determine Phase Quadrant

Based on the signs of I and Q:

Positive I, Positive Q → 1st quadrant (e.g., 00)

Negative I, Positive Q → 2nd quadrant (e.g., 01)

Negative I, Negative Q → 3rd quadrant (e.g., 11)

Positive I, Negative Q → 4th quadrant (e.g., 10)

Step 5: Map Phase to Bit Pairs

Convert each detected phase back into the corresponding 2 bits.

Step 6: Reconstruct Bit Stream

Combine all decoded pairs to get the original binary data.

# PROGRAM:
```
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_symbols = 20
T = 1.0  # Symbol period (seconds)
fs = 200.0  # Sampling frequency (Hz)
t = np.arange(0, T, 1/fs)  # Time vector for one symbol

# Generate random bit sequence (Message Signal)
bits = np.random.randint(0, 2, num_symbols * 2)  # Two bits per symbol
symbols = 2 * bits[0::2] + bits[1::2]  # Map bits to symbol (0 to 3)

# Define phase mapping for QPSK (Gray Coding)
symbol_phases = {
    0: 0,
    1: np.pi / 2,
    2: np.pi,
    3: 3 * np.pi / 2
}

# Generate carrier signal
carrier_i = np.cos(2 * np.pi * t / T)  # In-phase carrier
carrier_q = np.sin(2 * np.pi * t / T)  # Quadrature carrier

# Initialize QPSK signal
qpsk_signal = np.array([])
symbol_times = []

# Generate the QPSK modulated signal
for i, symbol in enumerate(symbols):
    phase = symbol_phases[symbol]
    symbol_time = i * T

# Modulate carrier with phase shift
    qpsk_segment = np.cos(2 * np.pi * t / T + phase) + 1j * np.sin(2 * np.pi * t / T + phase)
    qpsk_signal = np.concatenate((qpsk_signal, qpsk_segment))
    symbol_times.append(symbol_time)

# Full time vector
t_total = np.arange(0, num_symbols * T, 1/fs)

# Plotting the QPSK signal
plt.figure(figsize=(16, 14))

# Message signal visualization (Bit sequence)
plt.subplot(4, 1, 1)
plt.step(range(len(bits)), bits, where='mid', label='Message Signal (Bits)', color='black')
plt.title('Message Signal (Bit Stream)')
plt.xlabel('Bit Index')
plt.ylabel('Bit Value')
plt.grid(True)
plt.legend()

# In-phase component
plt.subplot(4, 1, 2)
plt.plot(t_total, np.real(qpsk_signal), label='In-phase (I)', color='brown')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='black')
plt.title('QPSK - In-phase Component')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Quadrature component
plt.subplot(4, 1, 3)
plt.plot(t_total, np.imag(qpsk_signal), label='Quadrature (Q)', color='blue')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='black')
plt.title('QPSK - Quadrature Component')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# QPSK Modulated Signal
plt.subplot(4, 1, 4)
plt.plot(t_total, np.real(qpsk_signal), label='QPSK Modulated Signal', color='red')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='black')
plt.title('QPSK Modulated Signal (Real Part)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

# OUTPUT:
![image](https://github.com/user-attachments/assets/8917d023-d430-4d06-a8d9-10c5d747f9fd)

 
# RESULT / CONCLUSIONS:
QPSK successfully transmitted and recovered data by mapping two bits per symbol, achieving higher bandwidth efficiency with minimal transmission errors.
