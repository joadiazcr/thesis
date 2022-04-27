import numpy as np
from numpy import pi
from matplotlib import pyplot

lat = 1e-6  # s
f_lp = 150e3  # Hz low-pass
f_zdb = 20e3  # Hz
f_ctlz = 5e3  # Hz controller zero
f_cavp = 16   # Hz cavity pole - cavity bandwidth

lp = 2*pi*f_lp      # /s low-pass
zdb = 2*pi*f_zdb    # /s
ctlz = 2*pi*f_ctlz  # /s controller zero
cavp = 2*pi*f_cavp  # /s cavity pole

f = 10**np.arange(0, 6, 0.01)
s = 2j*pi*f

A = zdb/cavp
print(A)
plant = 1/(1+s/cavp)
ctlr = A*(1+ctlz/s)/(1+s/lp)
loop = plant*ctlr

AdB = 20*np.log10(A)
print(AdB)
gap = 3
pyplot.figure(None, figsize=(10, 8), dpi=100)

pyplot.semilogx(f, f*0, color="gray")
pyplot.semilogx([f_ctlz, f_ctlz], [AdB+3, 90], color="gray")
pyplot.text(f_ctlz, 90+gap, "Controller zero")
pyplot.semilogx([f_zdb, f_zdb], [0, 80], color="gray")
pyplot.text(f_zdb, 80+gap, "0 dB crossing")
pyplot.semilogx([f_lp, f_lp], [AdB-3, 70], color="gray")
pyplot.text(f_lp, 70+gap, "Band limit")
pyplot.semilogx([f_cavp, f_cavp], [-3, 15], color="gray")
pyplot.text(f_cavp, 15+gap, "Cavity pole")

pyplot.semilogx(f, 20*np.log10(abs(plant)), label='Plant')
pyplot.semilogx(f, 20*np.log10(abs(ctlr)), label='Controller')
pyplot.semilogx(f, 20*np.log10(abs(loop)), label='Loop')

pyplot.xlabel("f (Hz)")
pyplot.ylabel("dB")
pyplot.xlim((2, 1e6))
pyplot.ylim((-90, 130))
pyplot.legend(frameon=False)
pyplot.show()
# pyplot.savefig("bode.png")
