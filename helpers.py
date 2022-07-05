from sklearn.metrics import accuracy_score
def print_accuracy(model,X_train,X_test,y_train,y_test):
    '''
    Prints the accuracy for the train and test sets after model fitting. 
    Parameters: model - model to score
    X_train - X_training set
    y_train - y_training set
    X_test - X_test set
    y_test - y_test set
    '''
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_acc = accuracy_score(train_preds, y_train)
    test_acc = accuracy_score(test_preds,y_test)
    print(f'Accuracy on the training set was {train_acc}')
    print(f'Accuracy on the test set was {test_acc}')
    print('\n')


def num_to_card(card_num):
    '''
    Reverses the encoding of the card number column. Number -> String
    '''
    if(card_num == 52):
        return 'Joker'
    else:
        ranks = ['2','3','4','5','6','7','8','9','10','Jack','Queen','King','Ace']
        suits = ['Diamonds','Clubs','Hearts','Spades']
        full = []
        for suit in suits:
            for rank in ranks:
                full.append(f'{rank} of {suit}')
    return full[card_num]