#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#resurses
#https://github.com/michael0419/TitleSkimmer/blob/main/app.py


# In[1]:


# Import pandas for data handling
import pandas as pd

# NLTK is our Natural-Language-Took-Kit
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Libraries for helping us with strings
import string
# Regular Expression Library
import re

# text vectorizers: CountVectorizer && TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# classifiers: MultinomialNB && RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


# Import some ML helper function
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report


# Import our metrics to evaluate our model
from sklearn import metrics
from sklearn.metrics import classification_report


# Library for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# You may need to download these from nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = stopwords.words('english')

#others
import plotly.express as px


# ## Load Data

# In[2]:


df = pd.read_csv('news_articles.csv')
print(df.shape)
df.head(5)


# ## "Target" a column where Real = 1, Fake = 0

# In[3]:


df['target'] = df.loc[:, 'label']
df = pd.get_dummies(df, columns=['target'], drop_first=True)
df.head(2)


# In[4]:


df.language.value_counts()


# In[5]:


#keeping only english language
df = df[df.language == 'english']


# In[6]:


df.language.value_counts()


# In[7]:


df.info()


# In[8]:


#  Inspect 
print(df.isnull().sum())


# In[9]:



df = df.dropna()


print(df.shape)


# In[10]:


df.isnull().sum()


# In[11]:


print(df.duplicated().sum())


# In[12]:


df = df.drop_duplicates()

print(df.shape, 'after')


# In[13]:


print(df.duplicated().sum())


# In[14]:


#  Find Label balances.
df.label.value_counts()


# In[15]:


#  Find Type balances.
df.type.value_counts()


# In[16]:


print("Original TEXT:", df['title'][10],":::REAL or FAKE:", df['label'][10])


# In[17]:


print("Clean TEXT from data:", df['title_without_stopwords'][10],":::REAL or FAKE:", df['label'][10])


# In[18]:


print("ORIGINAL TEXT:", df['text'][90])


# In[19]:


print("Clean TEXT from data:", df['text_without_stopwords'][90])


# In[20]:


colors = ['#808588','#d6cfc7']
plt.pie(df['label'].value_counts(), labels=['Fake','Real'],autopct="%0.2f", colors = colors)
plt.show()


# In[21]:


print(df.label.value_counts())
sns.countplot(x=df["label"])


# In[22]:


# colors = ['#808588','#d6cfc7']
print(df.type.value_counts())
plt.figure(figsize=(19,5))
sns.countplot(x= 'type', data= df, color='salmon', saturation = 0.1)


# In[23]:


#bs = all are fake news
plt.figure(figsize=(19,9))
sns.countplot(x= 'type', hue= 'target_Real', data= df)


# In[24]:


fig = px.sunburst(df, path=['label', 'type'])
fig.show()


# # Text Pre-Processing

# ### Stop words are already removed
# ### Punctuation are already removed
# ### Already in Lowercase

# In[25]:


#bring the root of the words
def root_words(string):
    porter = PorterStemmer()
    
    #  sentence into a list of words
    words = word_tokenize(string)
    
    valid_words = []

    for word in words:
        
        root_word = porter.stem(word)
        
        valid_words.append(root_word)
        
    string = ' '.join(valid_words)

    return string 

sent = 'I played and started playing with players and we all love to play with plays'
root_words(sent)


# In[26]:


def text_pipeline(input_string):
    input_string = root_words(input_string)
#     input_string= get_top_n_words(input_string)
    return input_string


# In[27]:


df['title_after'] = df['title_without_stopwords']
df['title_after'] = df['title_without_stopwords'].apply(text_pipeline)

print("Clean TEXT from data:", df['title_without_stopwords'][0])
print("CLEANDED TEXT:", df['title_after'][0])


# In[28]:


df['text_after'] = df['text_without_stopwords']
df['text_after'] = df['text_without_stopwords'].apply(text_pipeline)


# In[29]:


df.head(5)


# ## TOP real title and text

# In[30]:


from collections import Counter


# In[31]:


#title
real_news = []
for x in df[df['target_Real'] == 1]['title_after'].tolist():
    for word in x.split():
        real_news.append(word)


# In[32]:


len(real_news)


# In[33]:


# sns.barplot(pd.DataFrame(Counter(real_news).most_common(30))[0],pd.DataFrame(Counter(real_news).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()


# In[34]:


from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n = None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    freq_sorted = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return freq_sorted[:n]


top_unigram = get_top_n_words(df['text_without_stopwords'], 10)
words = [i[0] for i in top_unigram]
count = [i[1] for i in top_unigram]

plt.figure(figsize=(10,7))
plt.bar(words, count,align='center')
plt.xticks(rotation=90)
plt.ylabel('Number of Occurences')
plt.show()


# In[35]:


#title
df2 = pd.DataFrame(Counter(real_news).most_common(10))
df2


# In[36]:


# df2.groupby('word').sum()['count'].sort_values(ascending=False)
# fig=px.bar(df2,x='word',y='count',color='count',title='Top 10 bigrams')
# fig.show()


# In[37]:


#text
real_news2 = []
for x in df[df['target_Real'] == 1]['text_after'].tolist():
    for word in x.split():
        real_news2.append(word)


# In[38]:


#text
pd.DataFrame(Counter(real_news2).most_common(10))


# In[39]:


# sns.barplot(pd.DataFrame(Counter(real_news2).most_common(10))[0],pd.DataFrame(Counter(real_news2).most_common(10))[1])
# plt.xticks(rotation='vertical')
# plt.show()


# ## Top Fake title and text

# In[40]:


#title
fake_news = []
for x in df[df['target_Real'] == 0]['title_after'].tolist():
    for word in x.split():
        fake_news.append(word)


# In[41]:


#title
pd.DataFrame(Counter(fake_news).most_common(10))


# In[42]:


#text
fake_news2 = []
for x in df[df['target_Real'] == 0]['text_after'].tolist():
    for word in x.split():
        fake_news2.append(word)


# In[43]:


#text
pd.DataFrame(Counter(fake_news2).most_common(10))


# # Split the data into testing and training

# In[ ]:





# In[44]:


X = df['title_after'].values +' ' +  df['type'].values +  ' ' + df['text_after'].values 

y = df['label'].values


# In[45]:


print(df['title_after'].values)


# In[46]:


# Split our data into testing and training like always. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Save the raw text for later just incase
X_train_text = X_train
X_test_text = X_test


# In[47]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize our vectorizer
vectorizer = TfidfVectorizer()

# This makes your vocab matrix
vectorizer.fit(X_train)

# This transforms your documents into vectors.
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train.shape, type(X))


# In[48]:


features = vectorizer.get_feature_names()
weights = vectorizer.idf_

print(len(features), len(weights))

df_idf = pd.DataFrame.from_dict( {'feature': features, 'idf': weights})

df_idf = df_idf.sort_values(by='idf', ascending=False)

df_idf


# In[49]:


X_train[0]


# # Multinomial Naive Bayes

# In[50]:


# Initalize our model.
mn_model = MultinomialNB(alpha=.05)

# Fit our model with our training data.
mn_model.fit(X_train, y_train)

# Make new predictions of our testing data. 
y_pred = mn_model.predict(X_test)


# Make predicted probabilites of our testing data
y_pred_proba = mn_model.predict_proba(X_test)

# Evaluate our model
accuracy =  mn_model.score(X_test, y_test)

# Print our evaluation metrics
print("Model Accuracy: %f" % accuracy)


# In[52]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=mn_model.classes_))


# # Random Forest Classifier

# In[53]:


from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier()


# Fit our model with our training data.
rf_model.fit(X_train, y_train)


# Make new predictions of our testing data. 
y_pred = rf_model.predict(X_test)


# Make predicted probabilites of our testing data
y_pred_proba = rf_model.predict_proba(X_test)

# Evaluate our model
accuracy =  rf_model.score(X_test, y_test)

# Print our evaluation metrics
print("Model Accuracy: %f" % accuracy)

print(classification_report(y_test, y_pred, target_names=rf_model.classes_))


# # BernoulliNB
# 

# In[54]:


from sklearn.naive_bayes import BernoulliNB


# In[55]:


bnb = BernoulliNB()
bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)

accuracy =  bnb.score(X_test, y_test)
print("Model Accuracy: %f" % accuracy)
print(classification_report(y_test, y_pred, target_names=bnb.classes_))


# # ADABoostClassifier

# In[56]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[57]:


adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),n_estimators=5)
adab.fit(X_train,y_train)
y_pred3 = adab.predict(X_test)

accuracy =  adab.score(X_test, y_test)
print("Model Accuracy: %f" % accuracy)
print(classification_report(y_test, y_pred, target_names=adab.classes_))


# In[58]:


from sklearn.metrics import confusion_matrix


# In[59]:



cm = confusion_matrix(y_test, y_pred)

cm = cm.round(2)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

fig = plt.figure(figsize=(8,8))
ax = sns.heatmap(cm, annot=True, cmap='YlOrBr', fmt='g')
plt.title("Confusion Matrix of Real or Fake")
plt.xlabel('Predicted')
plt.ylabel('Actual')

print('true-negitive:', tn, 
      '\nfalse-positive:', fp, 
      '\nfalse-negative:', fn, 
      '\ntrue-positive:', tp )


# In[60]:


# UI = input("Enter a News Title: ")
# # 'NEWS RELEASE: DOH CITES SIX COMPANIES FOR AIR PERMIT VIOLATIONS'

# new_text = text_pipeline(UI)

# # print(new_text)

# new_text_vectorized = vectorizer.transform([new_text])

# #let user's guess the News type

# guess = input("Guess the News Type: ")

# if model.predict(new_text_vectorized)== 'Real' and guess == "Real":
#     print("Good job! You are Correct! \nYour News is: Real")

# elif model.predict(new_text_vectorized)== 'Fake' and guess == "Fake":
#     print("Good job! You are Correct! \nYour News is: Fake")
    
# elif model.predict(new_text_vectorized)== 'Fake' and guess == "Real":
#     print("OOPS! \nUnfortunately Your News is: Fake")

# elif model.predict(new_text_vectorized)== 'Real' and guess == "Fake":
#     print("OOPS! \nFortunately your News is: Real")


# In[62]:


#
UI = input("Enter a News Title: ")
# 'NEWS RELEASE: DOH CITES SIX COMPANIES FOR AIR PERMIT VIOLATIONS'

new_text = text_pipeline(UI)

# print(new_text)

new_text_vectorized = vectorizer.transform([new_text])
mn_model.predict(new_text_vectorized)


# In[63]:


new_text_vectorized


# In[63]:


import pickle
pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
pickle.dump(mn_model,open('mn_model.pkl','wb'))


# In[65]:


# pickle.dump(model,open('model.pkl','wb'))


# In[ ]:





# In[ ]:




