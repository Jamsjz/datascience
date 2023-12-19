# Password_@@123
# import streamlit as st
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
#
# # function to train the naive_bayes model
# def train_model(data, labels):
#     vectorizer = CountVectorizer()
#     X = vectorizer.fit_transform(data)
#     model = MultinomialNB()
#     model.fit(X, labels)
#     return model, vectorizer
#
# # function to predict spam or ham
# def predict(model, vectorizer, message):
#     X = vectorizer.transform([message])
#     prediction = model.predict(X)
#     return prediction[0]
#
# # streamlit app 
# def main():
#     st.title("Spam or Ham Detection")
#
#     # Trains the model
#     data = [
#             'hey, how are you?',
#             'Free money! Click here now!',
#             "I'm going to the park.",
#             "Congratulation! You've won a prize!",
#             "Remainder: Meeting tomorrow at 2 PM."
#             ]
#     labels = ["ham", "spam", "ham", "spam", "ham"]
#     model, vectorizer = train_model(data, labels)
#
#     # User Input
#     message = st.text_input ("Enter a message.")
#
#     # predict
#     if st.button("Predict"):
#         prediction = predict(model, vectorizer, message)
#         st.write("Prediction:", prediction)
#
# if __name__ == "__main__":
#     main()
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# Function to train the naive_bayes model and save it as a pickle file
def train_model(data, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    model.fit(X, labels)

    # Save the model as a pickle file 
    with open('spam_ham_model.pkl', 'wb') as file:
        pickle.dump((model, vectorizer), file)
    return model, vectorizer

def load_model():
    with open('spam_ham_model.pkl','rb') as file:
        model, vectorizer = pickle.load(file)
    return model, vectorizer

def predict(model, vectorizer, message):
    X = vectorizer.transform([message])
    prediction = model.predict(X)
    return prediction[0]

def main():
    st.title("Spam or Ham Detection")

    # Trains the model
    if not os.path.exists('spam_ham_model.pkl'):
        data = [
                'hey, how are you?',
                'Free money! Click here now!',
                "I'm going to the park.",
                "Congratulation! You've won a prize!",
                "Remainder: Meeting tomorrow at 2 PM."
                ]
        labels = ["ham", "spam", "ham", "spam", "ham"]
        model, vectorizer = train_model(data, labels)
    else:
        model, vectorizer = load_model()
    
    # User Input
    message = st.text_input ("Enter a message.")

    # predict
    if st.button("Predict"):
        prediction = predict(model, vectorizer, message)
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()

