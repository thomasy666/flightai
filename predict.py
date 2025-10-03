import sklearn.model_selection
import tensorflow as tf
import sklearn
import sys
import csv
import numpy
from sklearn.preprocessing import StandardScaler

EPOCHS = 10
CATEGORIES = 1
TEST_SIZE = 0.3

def main():
    #test to see the correct command line arguments
    if len(sys.argv) not in [2,3]:
        sys.exit('usage: python predict.py data.csv save.keras')
    flights, delay = load_data(sys.argv[1])
    
    flights = numpy.array(flights)
    delay = numpy.array(delay)
    scaler = StandardScaler()
    flights = scaler.fit_transform(flights)
    #seperate the data into training and testing data
    train_flights, test_flights, train_delay, test_delay = sklearn.model_selection.train_test_split(
        flights, delay, test_size=TEST_SIZE      
    )
    #create a model
    model = create_model()
    #train the model with the training data
    model.fit(train_flights, train_delay, epochs = EPOCHS)
    #
    model.evaluate(test_flights,  test_delay, verbose=2)


def load_data(filename):
    #load database into a list of strings, returns a list for each flight and a list for goal time to predict
    '''Load DAY_OF_MONTH,DAY_OF_WEEK,OP_UNIQUE_CARRIER,TAIL_NUM,DEST,DEP_DELAY,CRS_ELAPSED_TIME,DISTANCE,CRS_DEP_M,DEP_TIME_M,
    CRS_ARR_M,Temperature,Dew Point,Humidity,Wind,Wind Speed,Wind Gust,Pressure,Condition,sch_dep,sch_arr,TAXI_OUT '''
    
    flights = []
    delay = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for num, flight in enumerate(reader):
            each_flight = []
            each_flight.append(int(flight['DAY_OF_MONTH']))
            each_flight.append(int(flight['DAY_OF_WEEK']))
            each_flight.append(encode_carrier(flight['OP_UNIQUE_CARRIER']))
            each_flight.append(encode_airport(flight['DEST']))
            each_flight.append(int(flight['Temperature']))
            each_flight.append(int(flight['Dew Point']))
            each_flight.append(int(flight['Humidity']))
            each_flight.append(encode_wind(flight['Wind']))
            each_flight.append(int(flight['Wind Speed']))
            each_flight.append(int(flight['Wind Gust']))
            each_flight.append(float(flight['Pressure']))
            each_flight.append(int(flight['sch_dep']))
            each_flight.append(int(flight['sch_arr']))
            each_flight.append(encode_condition(flight['Condition']))
            flights.append(each_flight)
            delay.append(type_delay(int(flight['DEP_DELAY'])))
    return flights, delay
            

def encode_carrier(id):
    #returns an int for a carrier
    carriers = ['B6', 'DL', 'AA', 'AS', 'MQ', '9E', 'YX', 'HA', 'OO']
    for i, carrier in enumerate(carriers):
        if carrier == id:
            return i
    return -1
        
def encode_airport(id):
    #returns an int for a specific airport
    airports = ['CHS', 'LAX', 'FLL', 'MCO', 'ATL', 'ORD', 'BUF', 'LGB', 'LAS', 'DCA', 'PHX', 'SFO', 'SJU', 'SLC', 'BOS', 'SAV', 'SYR', 'MSP', 'SEA', 'MIA', 'PDX', 'TPA', 'BTV', 'IAH', 'DEN', 'RSW', 'ORF', 'JAX', 'MSY', 'CLT', 'BNA', 'RDU', 'SAN', 'SJC', 'ROC', 'DFW', 'IAD', 'AUS', 'DTW', 'PWM', 'SRQ', 'CMH', 'HNL', 'PBI', 'BWI', 'CLE', 'BUR', 'PIT', 'RIC', 'IND', 'CVG', 'SMF', 'ONT', 'SAT', 'PSP', 'OAK', 'ABQ', 'PSE', 'ORH', 'BQN', 'STT', 'RNO', 'PHL', 'EGE', 'JAC']
    for i, airport in enumerate(airports):
        if airport == id:
            return i    
    return -1
        
def encode_wind(dir):
    wind_directions = [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 
    'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 
    'W', 'WNW', 'NW', 'NNW'
]
    for i, directions in enumerate(wind_directions):
        if directions == dir:
            return i    
    return -1

def encode_condition(conditions):
    all_conditions = ['Fair / Windy', 'Fair', 'Light Rain / Windy', 'Partly Cloudy', 'Mostly Cloudy', 'Cloudy', 'Light Rain', 
                      'Mostly Cloudy / Windy', 'Partly Cloudy / Windy', 'Light Snow / Windy', 'Cloudy / Windy', 'Light Drizzle', 
                      'Rain', 'Heavy Rain', 'Fog', 'Wintry Mix', 'Light Freezing Rain', 'Light Snow', 'Wintry Mix / Windy', 
                      'Fog / Windy', 'Light Drizzle / Windy', 'Rain / Windy', 'Drizzle and Fog', 'Snow', 'Heavy Rain / Windy']
    for i, condition in enumerate(all_conditions):
        if condition == conditions:
            return i
    return -1

def type_delay(time):
    if time > 30:
        return 1
    else:
        return 0
        
def create_model():
    #creates a tensorflow model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(14, activation = 'relu', input_shape=(14,)),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(CATEGORIES, activation = 'sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics = ['accuracy'],
        loss = 'binary_crossentropy',
    )
    return model
        
if __name__ == '__main__':
    main()
