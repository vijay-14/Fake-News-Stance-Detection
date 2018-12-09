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
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

#definitions

def rem_stopwords(df_list):
    clean_list = []
    for row in df_list:
        clean_list.append([word for word in word_tokenize(row.lower()) if word not in stopwords.words('english')])
    return clean_list

def listtostring(lists):
    comb = [' '.join(row) for row in lists]
    return comb



#reading all files
train_bodies_files = pd.read_csv(sys.argv[1])
train_stances_files = pd.read_csv(sys.argv[2])
test_bodies = pd.read_csv(sys.argv[3])
test_stances = pd.read_csv(sys.argv[4])
print("read success")

#train remove stopwords
clean_headlines = rem_stopwords(train_stances_files.Headline.tolist())
clean_bodies = rem_stopwords(train_bodies_files.articleBody.tolist())
print("removed train stop words")

# test remove stopwords
test_clean_headlines = rem_stopwords(test_stances.Headline.tolist())
test_clean_bodies = rem_stopwords(test_bodies.articleBody.tolist())
print("removed test stopwords")

#combining all tokens to string
clean_headlines = listtostring(clean_headlines)
clean_bodies = listtostring(clean_bodies)
test_clean_headlines = listtostring(test_clean_headlines)
test_clean_bodies = listtostring(test_clean_bodies)

#converting train lists to dataframe
head_df = pd.DataFrame({'Body_ID': train_stances_files.Body_ID.tolist()})
head_df['Headline'] = clean_headlines
body_df = pd.DataFrame({'Body_ID': train_bodies_files.Body_ID.tolist()})
body_df['articleBody'] = clean_bodies

#converting test lists to dataframe
test_head_df = pd.DataFrame({'Body_ID': test_stances.Body_ID.tolist()})
test_head_df['Headline'] = test_clean_headlines
test_body_df = pd.DataFrame({'Body_ID': test_bodies.Body_ID.tolist()})
test_body_df['articleBody'] = test_clean_bodies

#combining Stances and Bodies datasets using "Bosdy_ID as key"
combined_data = pd.merge(head_df, body_df, on = 'Body_ID')
combined_test = pd.merge(test_head_df, test_body_df, on = 'Body_ID')

#concatinating Headlines and articleBody
merge_train = list(zip(combined_data.articleBody, combined_data.Headline))
merge2 = [row[0]+row[1] for row in merge_train]

#tag data for Doc2Vec model
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(merge2)]

#Doc2Vec Model

max_epochs = 20
vec_size = 1000
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

#train Doc2Vec vectors
train_vectors =[model.docvecs['{0}'.format(x)].tolist() for x in range(len(merge2))]
train_vectors = np.asarray(train_vectors)

# merge headline and body for test data
merge_test = list(zip(combined_test.articleBody, combined_test.Headline))
merge2_test = [row[0]+row[1] for row in merge_test]

#Predicting Doc vectors for test data
#tokenize test data first

test_data = []

for x in range(len(merge2_test)):
    tokenized_row = word_tokenize(merge2[x].lower())
    test_data.append(model.infer_vector(tokenized_row).tolist())

test_data = np.asarray(test_data)

#CNN model parameters
batch_size = 100
num_classes = 4
epochs = 30

#extracting train stances into array
target = [[rows] for rows in train_stances_files.Stance]
target = np.asarray(target)
y_train = keras.utils.to_categorical(target, num_classes)

#extracting test stances into array
test_target = [[rows] for rows in test_stances.Stance]
test_target = np.asarray(test_target)
y_test = keras.utils.to_categorical(test_target, num_classes)

#reshaping vectors for cnn
x = np.expand_dims(train_vectors, axis =2)
x_test = np.expand_dims(test_data, axis =2)

#CNN Model using Doc2Vec vectors

model = Sequential()

model.add(Conv1D(64,4, padding='same', input_shape=(1000,1)))
model.add(Activation('relu'))
model.add(Conv1D(32,4))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.50))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#CNN model Fit
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
test_actual=[row.tolist()[0] for row in test_target]

#Printing Results on test data
print("Confusion Matrix: \n", confusion_matrix(test_actual, f_preds))
print(classification_report(test_actual, f_preds), "Accuracy: ", accuracy_score(test_actual, f_preds))