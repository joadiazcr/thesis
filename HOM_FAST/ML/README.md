# ML training for HOM data collected at FAST (Fermilab)
This folder contains the Jupyter notebooks I wrote to train multiple NNs using HOM data collected at the FAST facility at Fermilab.
## The data
The data is save in the file **bpm_big_data_ex_std.plk**. It contains 17600 samples with 228 features.
## FAST_NN_training_A.ipynb
Simple example using a small portion of the data. I am doing this just to make sure the code makes sense. Filter data with bunch charge of 250 pC/b, 50 bunches, horizontal steering magnet at reference (H125=0.0) and up stream (US) HOM data.\
The filtered dataset has 1800 samples and 15 features.\
80-20 split for train and test datasets. Train dataset has 1440 samples. Test dataset has 360 samples.\
I am running the following kera models to predict `B441PV_std`:
- Linear regression single input (c1)
- Lineat regression multiple input (c1, ..., c8, V125)
- DNN single input (c1)
- DNN multiple input (c1, ..., c8, V125)

Steps of the training and result analysis
- Create and test normalization layer
- Build keras sequential model
- Configure the training of the model (.compile)
- Execute the training (.fit)
- Get the training process numbers (.history)
- Plot loss
- Evaluate the model using the test dataset (.evaluate)
- If single input: make predictions using the trained model and plot predictions vs input variable(.predict)
- Predict using test dataset and plot predictions vs labels (.predict)
- Plot test labels and test predictiones vs sample #

## FAST_NN_training_B.ipynb
Same as `FAST_NN_training_A.ipynb` except that `B` also uses down stream (DS) HOM data, therefore the dataset has 3600 samples, the train dataset has 2880 samples and the test dataset has 720 samples. Also, it only runs the following keras models to predict `B441PV_std`:
- Lineat regression multiple input (c1, ..., c8, V125, DS, US)
- DNN multiple input (c1, ..., c8, V125, DS, US)

## FAST_NN_training_C.ipynb
Same as `FAST_NN_training_B.ipynb` except that `C` also uses multiple values of H125 and bunch charges. Therefore, the dataset has 9600 samples, the train dataset has 7680 samples and the test dataset has 1920 samples. It runs the following keras models to predict `B441PV_std`:
- Lineat regression multiple input (c1, ..., c8, V125, DS, US, bunch charge, H125)
- DNN multiple input (c1, ..., c8, V125, DS, US, bunch charge, H125)