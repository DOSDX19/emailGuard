import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Initilizing whitelist/blacklist
whitelist = []
blacklist = []


#loading spam/ham emails senders dataset
senders_data = pd.read_csv('spam_email_dataset.csv', usecols=['Sender', 'Spam Indicator'], index_col='Sender')

#Loading spam/ham emails dataset
df = pd.read_csv('spam.csv')

data = df.where((pd.notnull(df)), '')

data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

X = data['Message'] 
Y = data['Category']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state  = 3)


feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()

model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Accuracy on training data: ', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('Accuracy on test : ', accuracy_on_test_data)

emails = []

message = input("Enter the message: ")
email_sender =  input("Enter the sender's email: ")

for email, spam_indecator in senders_data.iterrows():
    if email_sender == email and spam_indecator[len(spam_indecator) - 1] == 0:
        print(f'B >> {email} , {spam_indecator[len(spam_indecator) - 1]}')
        blacklist.append(email)
    else:
        print(f'W >> {email} , {spam_indecator[len(spam_indecator) - 1]}')
        whitelist.append(email)


emails.append(message)


email_data_features = feature_extraction.transform(emails)

prediction = model.predict(email_data_features)


print(prediction)

print('this sender is already in the blacklist' if email_sender in blacklist else 'this sender in the whitelist') 
print('Ham email' if prediction[0] == 1 else 'Spam email') 






