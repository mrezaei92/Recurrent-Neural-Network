import model
from keras.datasets import imdb
from keras.preprocessing import sequence

## load the dataset
top_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
word_to_index=imdb.get_word_index()
index_to_word = dict([(value, key) for (key, value) in word_to_index.items()]) 
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


### build the model
z=model.myLSTM(max_len=500)
z.build_model(vocabulary_size=top_words,embedding_size=32)


## Train the model

y=model.makeOneHot(y_train).T
z.train(X_train,y,num_epochs=5,droupout=0.5)


### evaluate the model

print("train accuracy= ",z.evaluate(X_train,y))
print("test accuracy= ",z.evaluate(X_test,model.makeOneHot(y_test).T))


