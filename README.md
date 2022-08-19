# Statistical Postprocessing of Local Numerical Weather Prediction Model Forecasts using Deep Learning

Accurate weather forecasts play a crucial role in many decision-making processes.
Currently, the main supply of weather forecasts is numerical weather prediction (NWP)
which solves physical equations to predict future states of the atmosphere. However,
the NWP models are prone to rapidly growing errors from the initial states, boundary
conditions, and model structures. In order to fix these systematic errors in the fore-
casts, statistical postprocessing methods are used. In this study, three alternative deep
learning architectures are proposed to statistically postprocess Global Ensemble Fore-
casting System (GEFS) forecasts of the Aegean Region of Turkey. The postprocessing
is done to multiple weather variables at multiple pressure levels. The input and output
structure of the models also introduced an extrapolation capability. The models are
trained with sixteen years of data, and the hyperparameters are tuned with one-year
validation data and tested over the last three years. The results are investigated from
the variable, pressure level, and location aspects. Fully convolutional and its U-Shaped
extension present promising results in every aspect. The U-Shaped architecture is cho-
sen over the others considering its lower mean, and lower variance in error distributions.
Also, the error distribution of extrapolated values validates the extrapolation capabil-
ity of the model. Finally, a case study on wind power forecasting of 19 power plants
shows that the method obtains better forecasts in a real-world application.
