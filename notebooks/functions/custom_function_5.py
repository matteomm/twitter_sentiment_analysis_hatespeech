#!/usr/bin/env python
# coding: utf-8

# In[26]:


# importing all the necessary libraries for the custom_functions
import re
import nltk

# for the lemmatizer function
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = nltk.stem.WordNetLemmatizer()

# for the word frequency function
from gensim.corpora import Dictionary
import itertools
from collections import defaultdict
import pandas as pd

# for the wordcloud function
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# for the Tweet Tokenizer
from nltk.tokenize import TweetTokenizer

# for the roc plot
import seaborn as sns

# for the performance function
from sklearn.metrics import accuracy_score,roc_auc_score, f1_score, recall_score
from sklearn.metrics import  roc_curve, confusion_matrix, precision_score



# In[27]:


def remove_pattern(input_txt, pattern):
    
    """ Function replacing a specific regex pattern with an empty space"""
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt    


# In[28]:


def counting(input_txt, pattern):
    
    """Simple function returning the pattern count instances in each tweet"""
    r = re.findall(pattern, input_txt)
    return len(r)


# In[29]:


def nltk_tag_to_wordnet_tag(nltk_tag):
    
    """ Function defining the actual part of speech as adjective, 
    verb, noun or adverb"""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# In[30]:


def lemmatize_sentence(sentence):
    
    """Function to lemmatize with POS all tweets"""
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
            # 'ass' kept being reduced to 'as' for some reason         
        if word == 'ass':
            lemmatized_sentence.append(word)
        
        elif tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


# In[31]:


tknz = TweetTokenizer()

def tokenize_tweet(s):
    """
    Tokenize each tweet into a list of words
    """
    tokens = tknz.tokenize(s)
    return [w for w in tokens]


# In[32]:


def tokenize_ngrams(s, ngram):
    """
    Tokenize each text into a list of words removing the ashtags in n-grams
    """
    tokens =  ngrams(s, ngram)
    return [w for w in tokens]


# In[33]:


def get_tokens_frequency_df(series):
    """
    Count each time the same word appeared in the series and returns a dataFrame
    """
    corpus_lists = [doc for doc in series.dropna() if doc]
    dictionary = Dictionary(corpus_lists)
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_lists]
    token_freq_bow = defaultdict(int)
    for token_id, token_sum in itertools.chain.from_iterable(corpus_bow):
        token_freq_bow[token_id] += token_sum

    return pd.DataFrame(list(token_freq_bow.items()), columns=['token_id', 'token_count']).assign(
        token=lambda df1: df1.apply(lambda df2: dictionary.get(df2.token_id), axis=1),
        doc_appeared=lambda df1: df1.apply(lambda df2: dictionary.dfs[df2.token_id], axis=1)).reindex(
        labels=['token_id', 'token', 'token_count', 'doc_appeared'], axis=1).set_index('token_id')


# In[34]:


def plot_word_cloud(df, top_n):
    
    """Creates a wordcloud based on term frequency of the first-n words"""
    word_cloud = WordCloud(background_color='white', colormap='magma', contour_width=1,
                           contour_color='orange', relative_scaling=0.5)

    sorted_freq_dict = dict(df[['token', 'token_count']].nlargest(top_n, columns='token_count').values)
    wc = word_cloud.generate_from_frequencies(frequencies=sorted_freq_dict, max_font_size=40)

    _, ax = plt.subplots(figsize=(15, 8))
    ax.set_title('Term Frequency', fontsize=16)

    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')


# In[35]:
def calculate_performance_1(model , data, target, name):
    
    """ Creates a mini dataframe with all info on the model performance"""
    
    predictions = model.predict(data)
    model_prob = model.predict_proba(data)[:,1]
    
    f1 = f1_score(target, predictions)
    accuracy = accuracy_score(target, predictions)
    roc_score = roc_auc_score(target, model_prob)
    precision = precision_score(target, predictions)
    nameit = str(name)
    score = pd.DataFrame()
    
    score['model'] = pd.Series(nameit) 
    score['f1'] = pd.Series(f1)
    score['accuracy'] = pd.Series(accuracy)
    score['roc_score'] = pd.Series(roc_score)
    score['precision'] = pd.Series(precision)
    
    return score

def calculate_performance(model , data, target):
    
    """ Creates a mini dataframe with all info on the model performance"""
    
    predictions = model.predict(data)
    model_prob = model.predict_proba(data)[:,1]
    
    f1 = f1_score(target, predictions)
    accuracy = accuracy_score(target, predictions)
    roc_score = roc_auc_score(target, model_prob)
    precision = precision_score(target, predictions)
    nameit = str(name)
    score = pd.DataFrame()
    
    score['model'] = pd.Series(nameit) 
    score['f1'] = pd.Series(f1)
    score['accuracy'] = pd.Series(accuracy)
    score['roc_score'] = pd.Series(roc_score)
    score['precision'] = pd.Series(precision)
    
    return score
    


# In[36]:


def plot_roc_curve(model, train, validation, y_train, y_val):
    
    """Plots the roc curves of two different sets"""
    
    base_pred_train = model.predict_proba(train)[:,1]
    base_fpr_train, base_tpr_train, base_thresh_train = roc_curve(y_train, base_pred_train)

    base_pred_validation = model.predict_proba(validation)[:,1]
    base_fpr_validation, base_tpr_validation, base_thresh_validation = roc_curve(y_val, base_pred_validation)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(12,7))
    ax1 = sns.lineplot(base_fpr_train, base_tpr_train, label='train',)
    ax1.lines[0].set_color("orange")
    ax1.lines[0].set_linewidth(2)

    ax2 = sns.lineplot(base_fpr_validation, base_tpr_validation, label='validaton')
    ax2.lines[1].set_color("yellow")
    ax2.lines[1].set_linewidth(2)

    ax3 = sns.lineplot([0,1], [0,1], label='baseline')
    ax3.lines[2].set_linestyle("--")
    ax3.lines[2].set_color("black")
    ax3.lines[2].set_linewidth(2)

    plt.title(f'{str(model)}', fontsize=20)
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPR', fontsize=16)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.text(x=0.8, y=0.8, s="50-50 guess", fontsize=14,
    bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.4"))

    plt.legend(loc=4, fontsize=17)
    plt.show();
    
def plot_roc_curve_1(model, train, validation, y_train, y_val, title):
    
    """Plots the roc curves of two different sets"""
    
    base_pred_train = model.predict_proba(train)[:,1]
    base_fpr_train, base_tpr_train, base_thresh_train = roc_curve(y_train, base_pred_train)

    base_pred_validation = model.predict_proba(validation)[:,1]
    base_fpr_validation, base_tpr_validation, base_thresh_validation = roc_curve(y_val, base_pred_validation)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(12,7))
    ax1 = sns.lineplot(base_fpr_train, base_tpr_train, label='train',)
    ax1.lines[0].set_color("orange")
    ax1.lines[0].set_linewidth(2)

    ax2 = sns.lineplot(base_fpr_validation, base_tpr_validation, label='validaton')
    ax2.lines[1].set_color("yellow")
    ax2.lines[1].set_linewidth(2)

    ax3 = sns.lineplot([0,1], [0,1], label='baseline')
    ax3.lines[2].set_linestyle("--")
    ax3.lines[2].set_color("black")
    ax3.lines[2].set_linewidth(2)

    plt.title(f'{title} roc_score', fontsize=20)
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPR', fontsize=16)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.text(x=0.8, y=0.8, s="50-50 guess", fontsize=14,
    bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.4"))

    plt.legend(loc=4, fontsize=17)
    plt.show();

# In[37]:


def max_seq_length(sequence):
    
    """Returns the the maximum length of word sequences created by the 
    keras tokenizer"""
    length = []
    
    for i in range(0, len(sequence)):
        length.append(len(sequence[i]))
    return max(length)


# In[ ]:

def clean_text(string):
    """Function makes the text columns similar 
       to preprocessing used for training model"""
    #removes retweets with http     
    new_str = re.sub(r"http://t(?!$)", '', string)
    #removes retweets with https
    new_str = re.sub(r"https?://[A-Za-z0-9./]*", '',new_str)
    #removes ashtags followed by numbers     
    new_str = re.sub(r"#[0-9]", '', new_str)
    # removes handles     
    new_str = re.sub(r"@[\w]*", '', new_str)
    new_str = re.sub(r"[^a-zA-Z'#]", ' ', new_str)
    return new_str




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




