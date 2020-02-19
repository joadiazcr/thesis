import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import control

def find_zero_crossing(vector, frequency_vector):
    """
    Find the 0-crossing given a vector in frequency domain and the associated frequency vector.
    """
    signs = np.sign(vector)
    signs[signs == 0] = -1
    zero_crossing = np.where(np.diff(signs))[0]
    return vector[zero_crossing], frequency_vector[zero_crossing], zero_crossing


def lpf(f, wc, cavity):
    gain = 20*np.log10(abs(cavity))
    phase = np.angle(cavity) * (180/np.pi)
    fc = wc/(np.pi*2)

    plt.subplot(2, 1, 1)
    plt.title("Low Pass Filter amplitude and phase response with cut-off frequency = %s" % fc)
    plt.plot(f, gain)
    plt.xscale('log')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.axvline(x=fc, color='r', linestyle='--')
    plt.ylabel('Gain [dB]')
    plt.xlabel('Frequency [Hz]')

    plt.subplot(2, 1, 2)
    plt.plot(f, phase)
    plt.xscale('log')
    plt.axhline(y=-45, color='r', linestyle='--')
    plt.axvline(x=fc, color='r', linestyle='--')
    plt.ylabel('Phase [degrees]')
    plt.xlabel('Frequency [Hz]')

    plt.show()


def pid(f, s, wc, cavity, ssa_bw = 53000, stable_gbw = 20000, control_zero = 5000, conf='nominal', plot = 0):
    # PI Controller Stability Analysis

    T = 1e-6  # Intrinsic Loop Delay # where is this number coming from?
    delay = np.exp(-s*T)

    plant = cavity*delay  # Plant (cavity + delay)

    # SSA modeled as a low pass filter
    low_pass_pole = 2*np.pi*ssa_bw
    lowpass = 1.0/(1.0+s/low_pass_pole)  # SSA noise-limiting Low-pass filter

    # Unity vector
    unity = np.ones(len(s))

    # Calculate corresponding values for ki and kp
    Kp = stable_gbw*2*np.pi/wc
    Ki = Kp*(2*np.pi*control_zero)
    print "Proportional gain kp=%s" % Kp
    print "Integral gain ki=%s" % Ki

    # Calculate transfer functions
    PI = (Kp + Ki/s)
    controller = PI*lowpass

    noise_gain = 1.0/(1+plant*controller)
    loop = plant*controller
    closed_loop = cavity/(1+plant*controller)

    # Gain and phase margin
    gain = 20*np.log10(abs(loop))
    phase = np.unwrap(np.angle(loop)) * (180/np.pi)
    gain_at_f_c, f_c, f_c_index = find_zero_crossing(gain, f)
    phase_at_f_180, f_180, f_180_index = find_zero_crossing(phase+180.0, f)
    GM = -gain[f_180_index]
    PM = 180 + phase[f_c_index]

    print "%s, %s, %s" % (gain_at_f_c, f_c, f_c_index)
    print "%s, %s, %s" % (phase_at_f_180, f_180, f_180_index)

    if plot == 1:
        plt.subplot(3, 1, 1)
        # Plot frequency response
        plt.semilogx(f, 20.0*np.log10(np.abs(plant)), linewidth="3", label='Plant')
        plt.semilogx(f, 20.0*np.log10(np.abs(controller)), linewidth="3",
                     label='Controller')
        plt.semilogx(f, 20.0*np.log10(np.abs(loop)), linewidth="3", label='Loop')
        plt.semilogx(f, 20.0*np.log10(np.abs(noise_gain)), linewidth="3",
                     label='Noise Gain')
        plt.semilogx(f, 20.0*np.log10(np.abs(unity)), '--', label='Unity')
    
        title = 'RF Station Analytical %s Configuration' % conf
        plt.title(title, fontsize=40, y=1.02)
        plt.ylabel('Magnitude [dB]', fontsize=30)
        plt.xlabel('Frequency [Hz]', fontsize=30)
        plt.legend(loc='upper right')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.rc('font', **{'size': 20})
    
        plt.subplot(3, 1, 2)
        plt.plot(f, gain)
        plt.xscale('log')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.ylabel('Gain [dB]')
        plt.xlabel('Frequency [Hz]')
        GM_text = 'GM = %.d dB' % int(GM)
        plt.annotate(GM_text, xy=(f_180*1.5, gain[f_180_index]))
        plt.annotate(s='', xy=(f_180, gain[f_180_index]), xytext=(f_180, unity[f_180_index]-1), arrowprops=dict(arrowstyle='<->'))
        f_c_text = r'$f_{\rm c} \approx %d\,{\rm kHz}$' % np.ceil(f_c*1e-3)
        plt.annotate(f_c_text,
                     xy=(f_c, 1), xytext=(f_c*1.2, 25), arrowprops=dict(facecolor='black', shrink=0.05))
    
        plt.subplot(3, 1, 3)
        plt.plot(f, phase)
        plt.xscale('log')
        plt.axhline(y=-180, color='r', linestyle='--')
        plt.ylabel('Phase [degrees]')
        plt.xlabel('Frequency [Hz]')
        PM_text = 'PM = %d$^{\circ}$' % int(PM)
        plt.annotate(PM_text, xy=(f_c*0.5, -250))
        f_180_text = r'$f_{\rm 180} \approx %d\,{\rm kHz}$' % int(round(f_180*1e-3))
        plt.annotate(f_180_text,
                     xy=(f_180, -180), xytext=(f_180*1.2, -100), arrowprops=dict(facecolor='black', shrink=0.05))
    
        plt.show()

    return plant, controller, loop, noise_gain, gain, phase 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Low Pass Filter and PID control concepts')
    parser.add_argument('-f', type=str, choices=['lpf', 'pid', 'swp', 'ir'],
                       help='Function: lpf (low-pass filter) or pid (pid controller) or swp (gain sweep) or ip (impulse response)')

    args = parser.parse_args()

    f = np.linspace(1, 1e6, 1e6)  # Frequency from 1 to 10^6 Hz
    s = 1j*2.0*np.pi*f  # Laplace operator s=jw=j*2*pi*f

    wc = 104.3  # rad/s. bandwidth? where is this number coming from? from 16.6 Hz cavity bandwidth
    cavity = 1.0/(1.0+s/wc)  # Cavity transfer function. Cavity acts as a low pass filter

    if args.f == 'lpf':
        lpf(f, wc, cavity)
    elif args.f == 'pid':
        configurations = ['nominal', 'high', 'hobicat']
        for conf in configurations:
            print "%s Configuration" %conf
            with open("%s.json" % conf, "r") as read_file:
                data = json.load(read_file)
            stable_gbw = data['stable_gbw']
            ssa_bw = data['ssa_bw']
            control_zero = data['control_zero']
            pid(f, s, wc, cavity, ssa_bw, stable_gbw, control_zero, conf, 1)
    elif args.f == 'swp':
        print "Proportional gain sweep"
        plt.subplot(2, 1, 1)
        conf = 'nominal'
	for ff in [0.1, 2, 3, 10]:
            with open("%s.json" % conf, "r") as read_file:
                data = json.load(read_file)
            stable_gbw = data['stable_gbw']*ff
            ssa_bw = data['ssa_bw']
            control_zero = data['control_zero']
            pid_results = pid(f, s, wc, cavity, ssa_bw, stable_gbw, control_zero, conf, 0)
            plt.semilogx(f, 20.0*np.log10(np.abs(pid_results[2])), linewidth="3", label='%s *k_p' %ff)
        unity = np.ones(len(s))
        plt.semilogx(f, 20.0*np.log10(np.abs(unity)), '--')

        print "Integral gain sweep"
        plt.subplot(2, 1, 2)
        conf = 'nominal'
        for ff in [0.1, 2, 3, 10]:
            with open("%s.json" % conf, "r") as read_file:
                data = json.load(read_file)
            stable_gbw = data['stable_gbw']
            ssa_bw = data['ssa_bw']
            control_zero = data['control_zero']*ff
            pid_results = pid(f, s, wc, cavity, ssa_bw, stable_gbw, control_zero, conf, 0)
            plt.semilogx(f, 20.0*np.log10(np.abs(pid_results[2])), linewidth="3", label='%s *k_i' %ff)
        unity = np.ones(len(s))
        plt.semilogx(f, 20.0*np.log10(np.abs(unity)), '--')

        plt.legend()
        plt.show()

    elif args.f == 'ir':
        print "Impulse response"
        T = np.arange(0, 0.5, 0.0001)  # Frequency from 1 to 10^6 Hz
        sys = control.tf2ss([1], [1/wc,1])
        T, yout = control.impulse_response(sys, T)
        #plt.plot(T, yout)

        print "Step response"
        cavity =  control.tf([1], [1/wc,1])
        kp = 10
        ki = 5
        controller = control.tf([kp, ki], [1,0])
        ol = cavity * controller
        sys = control.tf2ss(ol)
        T, yout = control.step_response(sys, T)
        plt.plot(T, yout)
        plt.show()

    else:
        lpf(f, wc, cavity)
        pid(f, s, wc, cavity)
