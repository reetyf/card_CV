    #!/usr/bin/env python
    # coding: utf-8
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as  plt
import glob
import playingcards
from skimage.transform import resize
import xml.etree.ElementTree as ET
from tqdm import tqdm
import warnings
def preprocess(img_x,img_y,deep,iterations = 2757):    
    warnings.filterwarnings('ignore')
    img_path = '/Users/Daniel/Desktop/brainstation/capstone/database/Images/Images/*'
    g1 = glob.glob(img_path)
    g1 = np.array(g1)

    annotation_path = '/Users/Daniel/Desktop/brainstation/capstone/database/Annotations/Annotations/*'
    g2 = glob.glob(annotation_path)
    g2 = np.array(g2)

    g1.sort()
    g2.sort()
    p = np.random.permutation(len(g1)) 
    g1 = g1[p]
    g2 = g2[p]

    card_img_arrays = []


    def suit_getter(suit):
            suit_num = np.NaN
            if(suit == 'S'):
                suit_num = 0
            elif(suit == 'C'):
                suit_num = 1
            elif(suit == 'H'):
                suit_num = 2
            else:
                suit_num = 3 
            return suit_num


    def card_maker():
        print('')
        
       
    count = 0
    for one_file in g1:
        count +=1
        if count == iterations:
            break;
        one = cv2.imread(one_file)
        one = cv2.resize(one, dsize = (img_x,img_y))
        card_img_arrays.append(one)


    card_class = []
    count = 0
    for one_file in g2:
        tree = ET.parse(one_file)
        root = tree.getroot()
        root = root[6][0]
        card_answer = root.text
        count +=1
        if count == iterations:
            break
        first_char = card_answer[0]
        second_char = card_answer[1]
        if(first_char == '1'): # first do 10s
            rank = 10
            third_char = card_answer[2]
            suit_num = suit_getter(third_char)
            card_to_add = playingcards.Card(rank,suit_num)
            card_class.append(card_to_add)
        elif(str.isdigit(first_char)): #2-9 done
            asint = int(first_char)
            suit_num = suit_getter(second_char)     
            card_to_add = playingcards.Card(asint,suit_num) 
            card_class.append(card_to_add)
        else: # only getss here if K,Q,J,A,Joker
            if(first_char == 'K'): # King
                suit_num = suit_getter(second_char)
                rank = 13 # King is rank 13
                card_to_add = playingcards.Card(rank,suit_num) 
                card_class.append(card_to_add)
            elif(first_char =='Q'): # queen
                suit_num = suit_getter(second_char)
                rank = 12 # Queen is rank 12
                card_to_add = playingcards.Card(rank,suit_num) 
                card_class.append(card_to_add)
            elif(first_char =='A'): # ace
                suit_num = suit_getter(second_char)
                rank = 1 # Ace is rank 1
                card_to_add = playingcards.Card(rank,suit_num) 
                card_class.append(card_to_add)
            elif((first_char =='J') & (second_char != 'O')): # if J and not O will decide on JOker functionality later
                suit_num = suit_getter(second_char)
                rank = 11 # Jack is rank 11
                card_to_add =playingcards.Card(rank,suit_num) 
                card_class.append(card_to_add)
            else:
                card_class.append('Joker') # joker placeholder

    card_img_arrays =np.array(card_img_arrays)
    img_df = pd.DataFrame()
    for one_img in card_img_arrays:
        img_df = img_df.append({"img_card_arr": np.array(one_img).flatten()}, ignore_index=True)
    card_class = np.array(card_class)
    card_df = img_df.copy()
    card_df['card_class'] = card_class

    card_df_pixels = card_df.copy()
    if(not deep):
        len_of_pixels = len(card_df['img_card_arr'][0])
        pixel_col_list = []
        for i in range(0,len_of_pixels):
            pixel_col_list.append(f'pixel_{i}')
        pixel_df = pd.DataFrame(index = range(0,card_df_pixels.shape[0]), columns = pixel_col_list)

        tqdm_loop = tqdm(range(0,card_df.shape[0]), desc= 'Loading...')
        for i in tqdm_loop:
            temp = card_df_pixels['img_card_arr'][i].tolist()
            for j in range(0,len_of_pixels):
                pixel_df.iloc[i,j] = temp[j]
        card_df_pixels = pd.concat([card_df_pixels,pixel_df],axis = 1)
    
    card_df_pixels['suit'] = np.NaN
    for i in range(0, card_df_pixels['card_class'].size):
        if(str(card_df_pixels.loc[i,'card_class']) == 'Joker'):
            card_df_pixels.loc[i,'suit'] = 'Joker'
        else:
            card_df_pixels.loc[i,'suit'] = card_df_pixels.loc[i,'card_class'].suit_name
    card_df_pixels['is_red'] = np.where((card_df_pixels['suit'] == 'Hearts') | (card_df_pixels['suit'] == 'Diamonds'),1,0)
    
    def suit_getter(suit):
        suit_num = np.NaN
        if(suit == 'Spades'):
            suit_num = 3
        elif(suit == 'Clubs'):
            suit_num = 1
        elif(suit == 'Hearts'):
            suit_num = 2
        elif(suit == 'Diamonds'):
            suit_num = 0
        else:
            suit_num = 4
        return suit_num
    card_df_pixels['suit_num'] = card_df_pixels['suit'].apply(suit_getter)
    
    
    
    card_df_pixels["card_string"]=card_df_pixels["card_class"].apply(str)

    def card_enumerator(card_string):
        if(card_string == 'Joker'):
            return 52
        else:
            ranks = ['2','3','4','5','6','7','8','9','10','Jack','Queen','King','Ace']
            suits = ['Diamonds','Clubs','Hearts','Spades']
            full = []
            for suit in suits:
                for rank in ranks:
                    full.append(f'{rank} of {suit}')
            return full.index(card_string)
    card_df_pixels['card_number'] = card_df_pixels["card_string"].apply(card_enumerator)
    return card_df_pixels


def view_head(pixel_df, x_dim,y_dim,num = 5):
    for i in range(0,5):
        plt.imshow(pixel_df.loc[i,'img_card_arr'].reshape(y_dim,x_dim,3))
        plt.title(str(pixel_df.loc[i,'card_class']))
        plt.show()


