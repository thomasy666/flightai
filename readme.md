Neural Network For Flight Delay

Background
Air traveling recovered quickly after the COVID-19 pandemic and is still one of the most popular transportation amongst the world. However, waiting to board the aircraft is always a stressful situation, especially when there is a delay. In this program, a Neural Network model is used to evaluate flights departing from John F Kennedy airport in New York. In this model the goal is to accurately categorize whether the flight will be delayed or not. 

A neural network will be used to create an ai that could make a prediction. The database includes domestic flights from 2018 in November and December. The goal is to accurately predict the variable DEP_DELAY which tells the number of minutes a flight is leaving the gate. A binary classification is used on DEP_DELAY, which identifies flights that leave after 30 minutes is delayed, otherwise it is considered as on time. 

Accuracy will be measured by the number of cases that the model gets correctly/ the total amount of cases. The loss function is a binary cross entropy. 

After running the model for 50 Epochs, the accuracy of the model reaches to about 0.93, and the loss reaches 0.23. This means that the model could accurately categorize the delay of the flight 93% of the time. The loss is calculated using the binary cross entropy, as it best fits the situation when trying to determine a binary catergory. 

The Neural network contains 4 different layers of neurons. The input is all converted into an integer or float representation to pass into the model. 

The neural network reads in 14 input values that could affect the departure time of the aircraft, including integers, floats, and objects.