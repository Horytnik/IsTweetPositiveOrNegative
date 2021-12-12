# IsTweetPositiveOrNegative
Simple application which detects if tweets are positive or negative. Data source and explanation:
http://help.sentiment140.com/for-students

In following application I used and compared Logisic Regression model and simple neural network model 
for positive/negative meaning of tweets. 

Done steps:
1. All words were calculated and printed top frequent for negative and positive.
2. From tweets were removed words which don't bring meaning like "the", Tweeter user names, links and etc.
3. Words were tokenized to allow classificators to learn.
4. Data was split to train and test.
5. Logistic Regression and Neural Network models were trained and tested. 