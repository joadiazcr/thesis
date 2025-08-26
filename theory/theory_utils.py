import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

class NCO:
    def __init__(self, f_clk, f_out, W, num_samples):
        self.f_clk = f_clk
        self.f_out = f_out
        self.W = W
        self.num_samples = num_samples

        f = Fraction(f_out) / f_clk
        f = f.limit_denominator()
        self.num, self.den = f.numerator, f.denominator

        # Get the int since we are in a digital system
        self.FCW = int((self.num / self.den) * 2**W)

    def plot(self):
        # NCO sine wave output
        t = np.arange(self.num_samples) * 1/self.f_clk
        sine_wave = np.sin(self.phase_norm)
        
        # Ideal output
        t_ideal = np.arange(self.num_samples * 100) * 1/(self.f_clk * 100)
        sine_ideal = np.sin(2 * np.pi * self.f_out * t_ideal)

        # error
        self.error_wave = sine_ideal[::100] - sine_wave

        # Plot
        plt.figure(figsize=(20,5))
        plt.plot(t, sine_wave, '--o', label='NCO')
        plt.plot(t_ideal, sine_ideal, label='Ideal')
        plt.plot(t, self.error_wave, label='error')
        plt.title("NCO / DDS Output (Phase Accumulator -> Sine)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

    def __repr__(self):
        str = (f"{self.__class__.__name__}:\n"
               f"  W={self.W}\n"
               f"  num={self.num}\n"
               f"  den={self.den}\n"
               f"  FWC={self.FCW}\n")
        return str
    

class NCO_simple(NCO):
    def __init__(self, f_clk, f_out, W, num_samples):
        super().__init__(f_clk, f_out, W, num_samples)

        phase_acc = 0
        self.phase_vals = [0]
        for _ in range(num_samples-1):
            phase_acc = (phase_acc + self.FCW) % (2**W)  # wrap at 2^W
            self.phase_vals.append(phase_acc)

        # Normalize phase to [0, 2π)
        self.phase_norm = (2 * np.pi / (2**W)) * np.array(self.phase_vals)


class NCO_complex(NCO):
    def __init__(self, f_clk, f_out, W, num_samples, L):
        super().__init__(f_clk, f_out, W, num_samples)
        self.L = L

        self.m = int(2**self.L / self.den)
        self.r = 2**self.W * self.num
        self.rem = self.r % self.den
        self.psl = self.rem * self.m
        self.modulo = 2**self.L % self.den

        phase_acc = 0
        phase_acc_low = 0
        offset = 0
        self.phase_vals = [0]
        for _ in range(num_samples-1):
            phase_acc_low = phase_acc_low + self.psl + offset
            if phase_acc_low >= (2**self.L):
                carry = (phase_acc_low & 2**self.L)/ (2**self.L)
                phase_acc_low = phase_acc_low % (2**self.L)
            else:
                carry = 0
            if carry:
                offset = self.modulo
            else:
                offset = 0
            phase_acc = (phase_acc + self.FCW + carry) % (2**self.W)  # wrap at 2^W
            print(f'phase_acc_low = {phase_acc_low}, carry = {carry}, offset = {offset}, pase_acc = {phase_acc}')
            self.phase_vals.append(int(phase_acc))

        # Normalize phase to [0, 2π)
        self.phase_norm = (2 * np.pi / (2**W)) * np.array(self.phase_vals)
        print(self.phase_vals)

    def __repr__(self):
        base_repr = super().__repr__()
        str = (f"{base_repr}"
               f"  L={self.L}\n"
               f"  psl={self.psl}\n"
               f"  modulo={self.modulo}\n")
        return str
