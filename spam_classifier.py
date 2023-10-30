import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Initilizing whitelist/blacklist
whitelist = list()
blacklist = list()


#loading spam/ham emails senders dataset
senders_data = pd.read_csv('spam_email_dataset.csv', usecols=['Sender', 'Spam Indicator'])
#extract senders from senders dataset
senders = senders_data[['Sender']]
#Convert sender column into series 
senders = pd.Series(senders.squeeze())

#Loading spam/ham emails dataset
df = pd.read_csv('spam.csv')

data = df.where((pd.notnull(df)), '')

data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1


# print(spam_emails)
# print(spam_emails)
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
    if email == email_sender and spam_indecator == 0:
        print(f'B >> {email} , {spam_indecator}')
        blacklist.append(email)
    else:
        # print(f'B >> {email} , {spam_indecator}')
        whitelist.append(email)

if email_sender in blacklist:
    print(f'{email_sender} was found in the blacklist')
    print(f'{email_sender} is spam email sender')
    exit()


emails.append(message)


# email = [""]

email_data_features = feature_extraction.transform(emails)

prediction = model.predict(email_data_features)


print(prediction)

print('this sender is already in the blacklist') #if  == 1 else 'Spam email') 
print('Ham email' if prediction[0] == 1 else 'Spam email') 






