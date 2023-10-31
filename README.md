# Labelling-time-series-data


In this data collection, we have a time series dataset that contains various measurements over time. Our main objective is to utilize one specific measurement, which is the speed recorded by GNSS sensors, to create labels for specific time windows. These labels will indicate whether a braking event occurred during that window.

After generating these labels, we plan to employ a supervised machine learning approach. This means we'll use historical data where we already know whether a braking event happened or not, and we'll use this information to train a machine learning model. 

Once the model is trained, we will evaluate its performance by testing it on new data. We want to see how accurately the other features in our dataset can predict whether a braking event occurred based on the patterns and information in the data.

In a sense, our goal is to use the speed data from GNSS sensors to mark when braking events happen, and then see how well our machine learning model can use other information (like accelerometer and Gyroscope) to predict these braking events.

## Machine learning

In the realm of traffic safety, braking detection, classification stands as the first step to identify hazardous events. By efficiently categorizing data, predicting potential hazards, and measuring the effectiveness of our predictions using the aforementioned metrics. Furthermore, the integration of advanced algorithms and sensor technologies enhances the precision of these predictions, thereby paving the way for more proactive and adaptive safety measures in vehicular systems.

