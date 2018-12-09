# agree =0
# disagree = 1
# discuss = 2
# unrelated = 3

import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from nltk import word_tokenize
from nltk.corpus import stopwords
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

#definitions

def row_count(train_stances,stance):
    count = 0
    for i in range(len(train_stances.Headline)):
        if train_stances_files.Stance[i] == stance:
            count+=1
    return count

def sampling(train_stances, stance,c1,c2):
    index = []
    for i in range(len(train_stances.Headline)):
        if train_stances.Stance[i] == stance:
            if random.random()<=(c1/c2):
                index.append(i)
    return index

def rem_stopwords(df_list):
    clean_list = []
    for row in df_list:
        clean_list.append([word for word in word_tokenize(row.lower()) if word not in stopwords.words('english')])
    return clean_list

def listtostring(lists):
    comb = [' '.join(row) for row in lists]
    return comb

def tfidfer(text_from_df):
    corpus = [rows for rows in text_from_df]
    vectorizer.fit(corpus)

def transform(text):
    corpus = [rows for rows in text]
    vectors = vectorizer.transform(corpus)
    print(vectors.toarray().shape)
    vectors = vectors.toarray()
    return vectors

#reading all files
train_bodies_files = pd.read_csv(sys.argv[1])
train_stances_files = pd.read_csv(sys.argv[2])
test_bodies = pd.read_csv(sys.argv[3])
test_stances = pd.read_csv(sys.argv[4])
print("read success")

#Counting the occurences for each category
agreed_count = row_count(train_stances_files,0)
disagreed_count = row_count(train_stances_files,1)
discuss_count = row_count(train_stances_files,2)
unrelated_count = row_count(train_stances_files,3)

#Extracting index for each category
agreed_indexes = sampling(train_stances_files,0,agreed_count,agreed_count)
discuss_indexes = sampling(train_stances_files,2,agreed_count,discuss_count)
unrelated_indexes = sampling(train_stances_files,3,agreed_count,unrelated_count)

disagreed_indexes = sampling(train_stances_files,1,agreed_count-disagreed_count,disagreed_count)
disagreed_indexes = disagreed_indexes+disagreed_indexes+disagreed_indexes+disagreed_indexes

#Selecting rows using the index for each category
agreed = train_stances_files.ix[agreed_indexes,:]
disagreed = train_stances_files.ix[disagreed_indexes,:]
discuss = train_stances_files.ix[discuss_indexes,:]
unrelated = train_stances_files.ix[unrelated_indexes,:]

#train remove stopwords
clean_headlines0 = rem_stopwords(agreed.Headline.tolist())
clean_headlines1 = rem_stopwords(disagreed.Headline.tolist())
clean_headlines2 = rem_stopwords(discuss.Headline.tolist())
clean_headlines3 = rem_stopwords(unrelated.Headline.tolist())

clean_bodies = rem_stopwords(train_bodies_files.articleBody.tolist())

# test remove stopwords
test_clean_headlines = rem_stopwords(test_stances.Headline.tolist())
test_clean_bodies = rem_stopwords(test_bodies.articleBody.tolist())

#train combine tokens to string
clean_headlines0 = listtostring(clean_headlines0)
clean_headlines1 = listtostring(clean_headlines1)
clean_headlines2 = listtostring(clean_headlines2)
clean_headlines3 = listtostring(clean_headlines3)
clean_bodies = listtostring(clean_bodies)

#test combine tokens to string
test_clean_headlines = listtostring(test_clean_headlines)
test_clean_bodies = listtostring(test_clean_bodies)

#train convert to data frame
head_df0 = pd.DataFrame({'Body_ID': agreed.Body_ID.tolist()})
head_df1 = pd.DataFrame({'Body_ID': disagreed.Body_ID.tolist()})
head_df2 = pd.DataFrame({'Body_ID': discuss.Body_ID.tolist()})
head_df3 = pd.DataFrame({'Body_ID': unrelated.Body_ID.tolist()})
head_df0['Headline'] = clean_headlines0
head_df1['Headline'] = clean_headlines1
head_df2['Headline'] = clean_headlines2
head_df3['Headline'] = clean_headlines3
body_df = pd.DataFrame({'Body_ID': train_bodies_files.Body_ID.tolist()})
body_df['articleBody'] = clean_bodies

#test convert to data frame
test_head_df = pd.DataFrame({'Body_ID': test_stances.Body_ID.tolist()})
test_head_df['Headline'] = test_clean_headlines
test_body_df = pd.DataFrame({'Body_ID': test_bodies.Body_ID.tolist()})
test_body_df['articleBody'] = test_clean_bodies

#combine headline and articleBody using Body_ID as key
combined_data0 = pd.merge(head_df0, body_df, on = 'Body_ID')
combined_data1 = pd.merge(head_df1, body_df, on = 'Body_ID')
combined_data2 = pd.merge(head_df2, body_df, on = 'Body_ID')
combined_data3 = pd.merge(head_df3, body_df, on = 'Body_ID')


combined_data = pd.merge(train_stances_files, train_bodies_files, on = 'Body_ID')
combined_test = pd.merge(test_head_df, test_body_df, on = 'Body_ID')

#Converting to list
merge_train0 = list(zip(combined_data0.articleBody, combined_data0.Headline))
merge_train1 = list(zip(combined_data1.articleBody, combined_data1.Headline))
merge_train2 = list(zip(combined_data2.articleBody, combined_data2.Headline))
merge_train3 = list(zip(combined_data3.articleBody, combined_data3.Headline))

merge_train = list(zip(combined_data.articleBody, combined_data.Headline))
merge_test = list(zip(combined_test.articleBody, combined_data.Headline))

#concatinating headline and articleBody texts
merge2_0 = [row[0]+row[1] for row in merge_train0]
merge2_1 = [row[0]+row[1] for row in merge_train1]
merge2_2 = [row[0]+row[1] for row in merge_train2]
merge2_3 = [row[0]+row[1] for row in merge_train3]

merge2 = [row[0]+row[1] for row in merge_train]
merge2_test = [row[0]+row[1] for row in merge_test]

#tfidf fit and transform
vectorizer = TfidfVectorizer(max_features=3000)
tfidfer(merge2_0)
t0 = transform(merge2)
test_t0 = transform(merge2_test)

vectorizer = TfidfVectorizer(max_features=3000)
tfidfer(merge2_1)
t1 = transform(merge2)
test_t1 = transform(merge2_test)

vectorizer = TfidfVectorizer(max_features=3000)
tfidfer(merge2_2)
t2 = transform(merge2)
test_t2 = transform(merge2_test)

vectorizer = TfidfVectorizer(max_features=3000)
tfidfer(merge2_3)
t3 = transform(merge2)
test_t3 = transform(merge2_test)

#combine train and test columns
arr_combined = np.column_stack((t0, t1, t2, t3))
test_arr_combined = np.column_stack((test_t0, test_t1, test_t2, test_t3))

#MLP model parameters
batch_size = 100
num_classes = 4
epochs = 200

#extract training rows
all_indexes = agreed_indexes + disagreed_indexes + discuss_indexes + unrelated_indexes
training_data = train_stances_files.ix[all_indexes,:]

#converting train stances to categorical
target = [[rows] for rows in training_data.Stance]
target = np.asarray(target)
y_train = keras.utils.to_categorical(target, num_classes)

#converting test stances to categorical
test_target = [[rows] for rows in test_stances.Stance]
test_target = np.asarray(test_target)
y_test = keras.utils.to_categorical(test_target, num_classes)

#train reshaping vectors for cnn
x = np.expand_dims(arr_combined, axis =2)
x = x[all_indexes,:]

#test reshaping vectors for cnn
x_test = np.expand_dims(test_arr_combined, axis =2)

#Multi-Layer Perceptron Model
model = Sequential()

model.add(Flatten())
#1st hidden layer
model.add(Dense(600, input_shape=(12000,1)))
model.add(Activation('relu'))
model.add(Dropout(0.35))
#2nd hidden layer
model.add(Dense(600))
model.add(Dropout(0.35))
#3rd hidden layer
model.add(Dense(600))
model.add(Dropout(0.35))
#softmax output layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#MLP model Fit
model.fit(x, y_train,
             batch_size=batch_size,
             epochs=epochs,
             validation_data=(x, y_train),
             shuffle=True)

#training performance
scores = model.evaluate(x,y_train, verbose=1)
print("Training - Loss, Accuracy: ",scores)

#test performance
scores = model.evaluate(x_test,y_test, verbose=1)
print("Testing - Loss, Accuracy: ",scores)

#test predictions
preds = model.predict(x_test, verbose = 1)

# converting predictions to stances
f_preds =[]
for row in preds:
    f_preds.append(np.where(row == row.max())[0].tolist()[0])

# converting actual predictions to list
test_actual=[row[0] for row in test_target]

print("Confusion Matrix: \n", confusion_matrix(test_actual, f_preds))
print(classification_report(test_actual, f_preds), "Accuracy: ", accuracy_score(test_actual, f_preds))