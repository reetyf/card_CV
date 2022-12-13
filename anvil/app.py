import cv2
import numpy as np
import joblib

model = joblib.load('rf.joblib')

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

import anvil.server
import anvil.media

anvil.server.connect("EU37GLRZWNOA6ZUIPHDTQMWY-XP5ORPH3VOQOXLPO")

@anvil.server.callable
def classify_image(file):
    with anvil.media.TempFile(file) as file:
        x_dim = 216
        y_dim = 288
        img = cv2.imread(file)
        img = cv2.resize(img, dsize = (x_dim,y_dim))
        img = img.flatten()
        img = img.astype('float')
        img /=255
        img= img.T.reshape(1,-1)
        pred = model.predict(img)
        pred = num_to_card(int(pred))
        maximum = np.amax(model.predict_proba(img))*100
        return (pred,round(maximum,2))

anvil.server.wait_forever()