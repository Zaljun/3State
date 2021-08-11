import radomForest as rf
import naiveBayes as nb
import KNN
import SVM
import decisionTree as dt

import pandas as pd

# get final dataset
final = pd.read_csv("realestatedata_final.csv")
final['Date'] = pd.to_datetime(final[['Year', 'Month']].assign(DAY=1))

# Insert news data to final
# News without sentiment
news_top = pd.read_csv("month_without_sentiment.csv")
news_top["Date"] = pd.to_datetime(news_top["Date"]) 

final_news_top = final.merge(news_top, left_on='Date', right_on = 'Date')

# News with sentiment
news_sen = pd.read_csv("month_with_sentiment.csv")
news_sen["Date"] = pd.to_datetime(news_sen["Date"])  

# Merge news into final
df_sen = final.merge(news_sen, left_on='Date', right_on = 'Date') # merge news with sentiment
df_top = final.merge(news_top, left_on='Date', right_on = 'Date') # merge news without sentiment

if __name__ == "__main__":
    cl = df_top.columns
    
    column1 = ['ZipCode', 'Unemp Rate']
    
    knn_m, knn_pred, knn_conf, knn_score = KNN.KNN(df_top, column1)
    print(knn_m)
    print(knn_conf)
    print(knn_score)
    
    knn_m[0].predict([['11554', '4.0']])