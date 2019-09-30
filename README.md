# Predict Bike Sales Using TensorFlow 2.0

![Bicycles](bicycles.jpg)

This project predicts bicycle sales using ANN (Artificial Neural Networks) using [Tensorflow](https://www.tensorflow.org) 2.0. 

### Data Reference:

This Hadi Fanaee-T Laboratory of Artificial Intelligence and Decision Support (LIAAD), University of Porto INESC Porto, Campus da FEUP Rua Dr. Roberto Frias, 378 4200 - 465 Porto, Portugal

### Data Description:

instant: record index
dteday : date
season : season (1:springer, 2:summer, 3:fall, 4:winter)
yr : year (0: 2011, 1:2012)
mnth : month ( 1 to 12)
hr : hour (0 to 23)
holiday : wether day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
weekday : day of the week
workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
weathersit :
1: Clear, Few clouds, Partly cloudy
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
temp : Normalized temperature in Celsius. The values are divided to 41 (max)
hum: Normalized humidity. The values are divided to 100 (max)
windspeed: Normalized wind speed. The values are divided to 67 (max)
casual: count of casual users
registered: count of registered users
cnt: count of total rental bikes including both casual and registered.


You can find the data in this directory. The file is: bike-sharing-daily.csv

### Step 1: Open a [Colab](https://colab.research.google.com) python notebook

### Step 2: Import TensorFlow and Python Libraries


```
!pip install tensorflow-gpu==2.0.0.alpha0
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### Step 3: Import the dataset

You will need to mount your drive using the following commands:
For more information regarding mounting, please check this out [here](https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory).


```
from google.colab import drive
drive.mount('/content/drive')
```

Upload the data file from Kaggle to your Google drive and then access it

```
bike = pd.read_csv('/content/drive/My Drive/Colab Notebooks/bike-sharing-daily.csv', encoding = 'ISO-8859-1')
```

Get more information about your dataset
```
bike
bike.info()
bike.describe()
bike.head(10)
bike.tail(10)
```

### Step 4: Visualize the dataset using Seaborn, a python library
See more steps in the colab.

### Step 5: Create testing and training data set and clean the data. 
See steps in the colab.

### Step 6: Train the Model. 
See steps in the colab.

### Step 7: Evaluate the Model. 
See steps in the colab.

### Step 8: Improve the Model
If you are not satisfied with the results, then you can increase the number of independent variables and retrain the same model. See steps in the colab.