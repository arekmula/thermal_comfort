# thermal_comfort
## Goal of the project
- predict temperature measured by temperature sensor in the next 15 minutes
- predict valve level of radiator in the next 15 minutes

## Data
The project was created in cooperation with [LERTA](https://www.lerta.energy/) and the data was shared by them. 
The data was registered in an office building located in Poznań. The registered data consisted of:
- room temperature from 3 different sensors in a room [&deg;C] 
- valve level of radiator [%]
- set temperature [&deg;C]
- timestamp for each measurement
- number of people in a room
- window direction

## Project requirements
- Python 3.8
- Listed in `requirements.txt`
- Input data for 1 prediction consisted of last 7 days before the prediction. 

## Project structure
```
thermal_comfort/
    common/ - common functions
    data/ - data for training and evaluation
    models/ - trained models
    processing/ - functions to predict values
    train/ - training code
    main.py - main code
```

To predict values run `main.py` script with 2 arguments.
```
python3 main.py input_file.json results.csv
```
- input_file.json - define time interval for prediction
- results.csv - file to save predicted values and ground truth values

## Results
Both values are predicted using Random Forest Regressor. Both models can be found in `models` directory

The results of a project was evaluated using Mean Average Error (MAE).

### Evaluation results
- room temperature: 0.057&deg;C
- valve level: 1.74 %

### Test results 
- room temperature: 0.068&deg;C
- valve level: 5.05 %
