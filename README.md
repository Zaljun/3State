# 3State (Predict price tendency of NY house)
Data Mining Project: to predict the price tendency using historical sales data, news data, government tax data, employment data, geographical data, crime data etc.

We used sales data, income data, employment rate, crime rate along with "scores" from news data. To calculate news scores, we simply used LDA Topic Modeling combined with Sentiment Analysis. 
Topic-sentiment monthly score: score = sum(doc(topics) * polarity) = sum([(topic,percentage)...] * polarity)

Most data sources are from zillow sales data, US government, and Kaggle news data (from 2010 - 2019) https://www.kaggle.com/snapcrack/all-the-news?select=articles1.csv 
