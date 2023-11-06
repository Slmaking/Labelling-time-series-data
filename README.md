# Labelling-time-series-data


In this data collection, we have a time series dataset that contains various measurements over time. Our main objective is to utilize one specific measurement, which is the speed recorded by GNSS sensors, to create labels for specific time windows. These labels will indicate whether a braking event occurred during that window.

After generating these labels, we plan to employ a supervised machine learning approach. This means we'll use historical data where we already know whether a braking event happened or not, and we'll use this information to train a machine learning model. 

Once the model is trained, we will evaluate its performance by testing it on new data. We want to see how accurately the other features in our dataset can predict whether a braking event occurred based on the patterns and information in the data.

In a sense, our goal is to use the speed data from GNSS sensors to mark when braking events happen, and then see how well our machine learning model can use other information (like accelerometer and Gyroscope) to predict these braking events.

## Machine learning

In the realm of traffic safety, braking detection, classification stands as the first step to identify hazardous events. By efficiently categorizing data, predicting potential hazards, and measuring the effectiveness of our predictions using the aforementioned metrics. Furthermore, the integration of advanced algorithms and sensor technologies enhances the precision of these predictions, thereby paving the way for more proactive and adaptive safety measures in vehicular systems.

In the process of segmenting time series data for vehicle maneuvers, it's essential to capture the full context of each maneuver. For example, a hard braking incident may span just a few seconds, while a lane change could take up to 30 seconds. The overarching objective of this segmentation strategy is to ensure that events like hard braking or lane changes are thoroughly captured within a windowed segment. Drawing inspiration from the work of Carlos et al., they found the optimal length for detecting road anomalies such as potholes or speed bumps to be 30 data points, equating to 3 seconds at a 10 Hz frequency. This principle is evident in Figure 17, where a noticeable spike in acceleration surpasses a set threshold. To categorize such a moment as a specific traffic event, we incorporate an additional 30 data points both preceding and following the spike. This inclusion ensures a holistic view of the event, preventing the potential misinterpretation of an isolated peak.
Further elaborating on our methodology described in the manuscript, we utilize a sliding window approach with a keen emphasis on preventing overlaps, especially for segments categorized as normal driving. In situations where overlaps are probable, we have set clear hierarchies for events to eliminate ambiguities. For the modeling phase, specifically with the decision tree, we opted for inputs based on 1Hz data points rather than variable-length windows. This decision not only facilitates a standardized input format but also augments the model training process, thereby solidifying the robustness and clarity of our methodology.

  <img src="https://github.com/Slmaking/Labelling-time-series-data/blob/f42aa25fb5d05db92c1a831fec748c7f33acd39c/code/Screenshot%202023-10-29%20at%2018.33.43.png" alt="QR">



