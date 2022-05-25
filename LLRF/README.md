# LLRF
## cavity_model.py
  This code is intended to create functions for each component of an RF station. To proof the functions, the code can reproduce tests and plots of Chapter 3 "RF Station" of the "LCLS-II system simulations: Physics" Document. Includes tests for:
  - Cavity response to a step on drive signal.
  - Cavity response to step on beam current.
  - Cavity response to a step on drive signal under several detuning frequencies.
  - PI loop stability analysis for 3 gain configurations: nominal, hobicat and high gain. Uses the `pid` function of `PIcontrol.py`.
  - SSA model test
  - Phase shifft
  - Gaussian noise

  The document is available at https://beamdocs.fnal.gov/AD/DocDB/0049/004978/001/physics.pdf

## cavity_step_response.py
  Uses the RF component functions of `cavity_model.py` to implement a model of an RF station in the function `cavity_step`. This RF station model is used to simulate an RF station under different conditions. Feedforward, detuning, measurement noise and beam can be enable/disable to evaluate their behaviour. 

## pi_gain_analysis.py
  Optimize proportional and integral gains of the LLRF controller by minimizing the integral of the cavity voltage error. Creates `simulation_Ymd_HMS` directories with 2 files: `data.npy` which contains the results of the simulations and `simulation_settings.json`. 