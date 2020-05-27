
# Twitter sentiment analysis to detect hate speech
## *Viewers are warned that this page contains content that is offensive and objectionable*
<p align="center">
  <img src="https://github.com/matteomm/twitter_sentiment_analysis_hatespeech/blob/master/figures/cover.jpg" width=750>
</p>


Different tweets from two sources were grouped in order to build a larger collection of roughly 40000 tweets contaning positive, neutral and offensive language. Offensive language is also divided into simple offensive language and hate speech.
Hate speech's definition is taken from Cambridge Dictionary: "public speech that expresses hate or encourages violence towards a person or group based on something such as race, religion, sex, or sexual orientation". The main goal of this project is to build a model capable of identifying hate speech on Twitter. In the final section winning model runs across fresh tweets collected twice daily from Twitter API in the UK and US, showing percentage of overall tweets labelled as offensive language or hate speech. **Fresh updates and insights can be found on the online Heroku app *[here](https://dashboard.heroku.com/apps/shrouded-sands-52273/resources)**.

Promptly identifying hate speech and overall offensive language is a very important concern for all major social media platforms. According to the public order act from 1986, any communication which is threatening or abusive, and is intended to harass, alarm, or distress someone is forbidden. The penalties for hate speech include fines, imprisonment, or both. It is very important to ensure social platform media only act as a medium of spreading free speech and constructive ideas. One of the main challenges when it comes to hate speech detection is also to differentiate it from any other type of offensive language.


Contacts:
* [e-mail](matteotortella4@gmail.com)
* [Linkedin](https://www.linkedin.com/in/matteo-tortella-0a4274130/)


# Table of Contents

1. [ File Descriptions ](#file_description)
2. [ Technologies Used ](#technologies_used)
3. [ Executive Summary ](#executive_summary)
    * [ Data Cleaning and Feature Engineering ](#datacleaning)
    * [ Exploratory Data Analysis ](#eda)
    * [ Modelling ](#modelling)
    * [ Twitter API and MySQL Storage ](#twitterapi)
    * [ Model Evaluation and Dashboard ](#insights)
 4. [ Limitations and Future Work ](#futurework)


<a name="file_description"></a>
## File Descriptions
> ipynb_checkpoints: different notebooks version going from preprocessing to modelling

> notebooks: contains all the different notebooks used throughout the project from data cleaning to final app

> data: contains dataset used for the analysis both processed and raw

> references: links to the source material referenced in the notebook

> figures: jpg images taken from the jupyter notebook 

> functions: folder with all custom functions created and called onto notebooks 

> twitter_presentation: pdf format of a presentation with key insights for non-tecnhical stakeholders

<a name="technologies_used"></a>
## Technologies used
- Python
- Pandas
- Keras
- Nltk
- Tweepy
- MySQL
- Seaborn
- Matplotlib
- Streamlit
- Heroku

<a name="executive_summary"></a>
## Executive Summary

Cyber bullying and aggressive language on social plaforms are one of the plagues of our modern time. Freedom of speech online can easily derail into offensive, unjustified and unconstuctive criticism towards sexual, political and religious beliefs.
ML classifiers and the wealth of data available on these platforms offer a valid solution in order to mitigate this issue.

In this project, a series of classifiers such as Logistic Regression, Decision Trees and CNN were trained on 40000 thousand tweets human labelled as offensive and not offensive. The 40000 tweets were assembled by combining two different sets. One of them was originally taken from an [Analytics Vidhaya](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/) competition while the second dataset was a collection of 20000 offensive tweets found on [Github](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data).

After the initial preprocessing, the first section is focused on training a classifier at recognising hate speech. The final winning model was Logistic Regression with a final f-1 score of **98.2%** on validation set.

The second section concentrates more on using the refined model to make predictions on unseen Tweets freshly taken from the Twitter API and showcasing out findings on a web app deployed on Heroku.


<a name="datacleaning"></a>
## Data Cleaning and Feature Engineering

As mentioned above the initial dataset was designed through the combination of two distinct sets from the web. The raw initial dataset was designed to have no class imbalance as we have selected exactly 21421 positive and 20610 negative tweets. Given the balanced nature of the training set, this project will mainly look at accuracy and f1 score as success metrics. 

Cleaning was performed with a some iterations of regex syntax to get rid of re-tweets, handles and special characters. Duplicate tweets were also removed and lemmatization with part of speech was also applied to all tweets. The last stage involved removing stopwords and also words shorter than three characters as they do not usually carry very valuable meaning. However, stopwords such as 'but' and 'not' were kept for neural networks sequencing. We created two additional columns, 'tweet_length' and 'handle_count' to investigate whether these two factors have any impact on positive/negative language.

<a name="eda"></a>
## Exploratory Data Analysis

The EDA section provided some useful insights into the very fabric of the words used in these tweets. Wordclouds were created to showcase the most common 1,2 and 3-grams present in the text. Attaching below a comparison between positive and negative lexicon, larger words correspond to higher frequency.

<p align="center">
  <img src="https://github.com/matteomm/twitter_sentiment_analysis_hatespeech/blob/master/figures/word_clouds.png" width=750>
</p>

Also, distributions of lengths between positive and negative tweets was also analysed and, as shown in the graph below, negative tweets seem to be on average shorter than their positive counterpart. 1 represents negative tweets while 0 positive ones. It is possible to see that most of the negative tweets are concentrated on the left side of the graph corresponding to shorter lengths. A simple t-test confirmed that the mean difference is significant with p-value smaller than 0.001.

<p align="center">
  <img src="https://github.com/matteomm/twitter_sentiment_analysis_hatespeech/blob/master/figures/length.png" width=750>
</p>

In the last section, the relationship between number of handles and aggressiveness was measured by plotting again number of positive/negative against overall number of handles. The vast majority of the tweets had somewhere in between 0 and 3 handles with a stark difference between 0 and 1 handles tweets, the latter having a significantly higher proportion of offensive tweets along with the 2 and 3 class. This could be explained by people directed their rant at someone through the use of handles.

<p align="center">
  <img src="https://github.com/matteomm/twitter_sentiment_analysis_hatespeech/blob/master/figures/handles.png" width=750>
</p>

Are aggressive people less verbose? Add final graph

<a name="modelling"></a>
## Modelling

The only predictor I used for the modelling was only the text itself. The lemmatized and refined version of our text was vectorized with the tf-idf method. Tf-idf was preferred over Bag-of-words as words rarity is quite relevant in this instance.

The tf-idf matrix was used across all models except for CNN (which only takes as input a tokekized sequence of words) and Naives Bayes where I also tried to use a Bow framework just to see whether performance would take a hit.

CNN performance was not added to the graph below, however it was very low with accuracy score just above 50%.
More work is needed on the Neural Network section where I could potentially look to implement RNN on top of the low performing CNN already in place.

All models were tweaked and optimised with GridSearch CV. As mentioned earlier on, the final best performing model was a Logistic Regression with a **98.2% f-1 score on the validation set**. Attaching below a snapshot of all models and iterations ran on the notebook across training and validation. 

<p align="center">
  <img src="https://github.com/matteomm/twitter_sentiment_analysis_hatespeech/blob/master/figures/models.png" width=750>
</p>

Logistic Regression comes also quite handy at describing how each word is important at predicting outcomes. 
Below we can see the top 20 most important word coefficients in order to deem something as offensive:

<p align="center">
  <img src="https://github.com/matteomm/twitter_sentiment_analysis_hatespeech/blob/master/figures/logistic_coefficients.png" width=750>
</p>


Although Decision Tree and Naive Bayes had reasonably good accuracy and f-1 scores score in the validation set, Logistic Regression was still the best and it was also preferred for higher interpretability with **final accuracy score of 98.7% on the test set**. The winning model and the tf-idf were both pickled for later use. Please find below stats for the test set.

<p align="center">
  <img src="https://github.com/matteomm/twitter_sentiment_analysis_hatespeech/blob/master/figures/winning_model_test.png" width=750>
</p>


<a name="twitterapi"></a>
## Twitter API and MySQL Storage 

After requesting access to the Twitter Developer portal, fresh tweets have been collected periodically on a local MySQL database which has been created only to accomodate the incoming stream of tweets. The python library Tweepy was used to create a connection with the Twitter API. The only information we are storing on the SQL database from the raw JSON stream are: 

> 1. Twitter ID
> 2. Time of Tweet
> 3. Twitter Text

All text had to be stripped of emojis in order to be stored in the database.
The stream listener of Tweepy can collect information based on specific topics and other filters such as language.
In this case, all words with the ashtag 'coronavirus' were tracked to ensure a high volume of tweets streaming. At the same time, only the english language was selected for convenience.

<a name="insights"></a>
## Model Evalutation and Dashboard
Lastly, a basic pipeline was engineered by pulling batches of newly streamed Tweets on a separate notebook where exactly the same preprocessing cleaning was applied. In this section, the final model and the tf-idf pickled objects are recalled. The only fitted tf-idf object is transformed onto the new tweets so that a matrix with each time exactly the same amount of columns (9000) is generated. Out of words vocabulary are just dropped in this case while the rest is retained by the matrix.

Subsequently, the predict_proba function is applied to the vectorized matrix and only values above .8 are filtered in the hope of collecting only tweets that the model deems as very offensive. As a last step, fluctuations over time of offensive tweets and most recurring words are plotted. This figure is then uploaded onto Heroku through Streamlit. The final app can be found here.

<a name="futurework"></a>
## Limitations and Future Work
Although the final model performance is very good even on the test results of our dataset, one of the main limitations of this project is measuring the performance of the model on fresh tweets. Pragmatically, it is possible to just look at some of the tweets labelled as negative and subjectively consider them as offensive or not. This very last point raises another important issue in this framework which is the one related to the inherent bias of human people manually labelling tweets. 

The judgment used to label the initial ground truths is also fallacious as what is offensive for some people might not be offensive for others. Out of words vocabulary is probably one of the main drawbacks of this model. The algorithm does not deal well with sentences that have loads of words not contained in our initial 9000 thousand word long vocabulary. Recursive Neural Network would probably be the best alternative when it comes to deal with out of vocabulary words. 
