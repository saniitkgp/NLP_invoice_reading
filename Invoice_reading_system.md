```python
#------------------------------importing library-----------------------------------------
from dateutil.parser import parse
import numpy as np
import pandas as pd
from dateutil.parser import parse
import pickle
from bpemb import BPEmb
import pytesseract
import cv2
import glob
import matplotlib.pyplot as plt
from PIL import Image
from bs4 import BeautifulSoup
import re
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch.utils.data import Dataset 
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from scipy.sparse import csr_matrix
from torch_geometric.utils import convert

import torch.nn as nn
from timeit import default_timer as timer
```


```python
def SaveData(data,path):
    with open(path,'wb') as file:
        pickle.dump(data,file)

def LoadData(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


# boolean_feature =['isDate','isZipCode','isKnownCity','isKnownDept','isKnownCountry', # below is nature feature
# 'isAlphabetic','isNumeric','isAlphaNumeric','isNumberwithDecimal','isRealNumber','isCurrency','hasRealandCurrency','mix/mixc']

def is_date_o(string, fuzzy=False):
    print(string)
    try: 
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False
def is_date(string, fuzzy=False):
#     print(string)
    try:
        if is_number(string):
            if(int(string) > 9999):
                return False
        elif is_numberwithDecimal(string):
            return False     
    except:
        if is_numberwithDecimal(string):
            return False
    try: 
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False
    
def is_zipcode(code_str,database):
    if is_number(code_str):
        if code_str in database:
            return True
        else:
            return False
    else :
        return False
    
def is_knowCity(city_str,database):
    if is_alphbetic(city_str):
        if city_str in database:
            return True
        else:
            return False
    else :
        return False

def is_knownDept(dept_str,database):
    if is_alphbetic(dept_str):
        if dept_str in database:
            return True
        else:
            return False
    else :
        return False

def is_knownCountry(country_str,database):
    if is_alphbetic(country_str):
        if country_str in database:
            return True
        else:
            return False
    else :
        return False    
    
# creatring fucntion for nature vector
def is_alphbetic(alph_str):
    return bool(re.match('^[a-zA-Z]+$', alph_str))

def is_number(num_str):
    return bool(re.match('^[0-9]+$', num_str))

def is_alphaNumeric(alphNum_str):
    return bool(re.match('^[a-zA-Z0-9]+$', alphNum_str))

def is_numberwithDecimal(num_str):
    
    if is_realNumer(num_str):  # checking if it is in numeric form
        num = num_str.split('.') 
        if len(num)>=2 and len(num[-1])>=1: # length after decimal point must be >=1
            return True
        else:
            return False
    else:   
        return False

    
def is_realNumer(num_str):
    try: 
        float(num_str)
    except ValueError: 
        return False
    return True
    
def is_currency(curr_str):
    curr_symbol='¥$€£₹'
    curr_str =list(curr_str)
    if curr_str[0] in curr_symbol or curr_str[-1] in curr_symbol:
        return True
    else:
        return False
    
def is_hasRealandCurrency(curr_str):
    currency = list(curr_str)
    curr_nature ='-+'
    if currency[0] in curr_nature:
        if is_currency(''.join(currency[1:])):
            return True
        else:
            return False
    else:
        return False
        

def create_nature_vector(word):
    nature=[] # 8 -dimentional vector 
    nature.append(is_alphbetic(word))
    nature.append(is_number(word))
    nature.append(is_alphaNumeric(word))
    nature.append(is_numberwithDecimal(word))
    nature.append(is_realNumer(word))
    nature.append(is_currency(word))
    nature.append(is_hasRealandCurrency(word))    
    if not any(nature):
        mix =True
    else:
        mix =False
    nature.append(mix)
    return nature 

def create_Boolean_feature (word,database):
    # boolean_feature =['isDate','isZipCode','isKnownCity','isKnownDept','isKnownCountry', # below is nature feature
    # 'isAlphabetic','isNumeric','isAlphaNumeric','isNumberwithDecimal','isRealNumber',
    #'isCurrency','hasRealandCurrency','mix/mixc']

    date = is_date(word)
    zipcode = is_zipcode(word,database[0])
    city = is_knowCity(word,database[1])
    dept = is_knownDept(word,database[2])
    country = is_knownCountry(word,database[3])
    nature_vect = create_nature_vector(word) 
#    print(['isDate','isZipCode','isKnownCity','isKnownDept','isKnownCountry', # below is nature feature
#           'isAlphabetic','isNumeric','isAlphaNumeric','isNumberwithDecimal','isRealNumber',
#    'isCurrency','hasRealandCurrency','mix/mixc'])
    return np.array([date,zipcode,city,dept,country]+nature_vect)
    





#https://github.com/bheinzerling/bpemb/blob/master/bpemb/bpemb.py
def create_text_feature(word, bpemb_encoder):
    ids = bpemb_encoder.encode_ids(word)
#    print(ids)
    if len(ids)==1:
        embding = bpemb_encoder.vectors[ids]
        embding =np.concatenate((embding.flatten(), np.zeros((200))))
    elif len(ids)==2:
        embding = bpemb_encoder.vectors[ids]
        embding =np.concatenate((embding.flatten(), np.zeros((100))))
    elif len(ids)==3:
        embding = bpemb_encoder.vectors[ids]
        embding=embding.flatten()
    else:
#        embding = bpemb_encoder.vectors[ids]  # need to take care of case having subword length >3
#        embding = np.split(embding, 4)[0:3]
         embding = np.zeros((300))   
#    print(len(embding))
    return embding


def details(data):
    print('data type :',type(data))
    try:
        print('data shape : ', data.shape)
    except:
        print('data lenght : ',len(data))

def create_feature_vector(word_dict,num_feature):
     database = LoadData(database_path) #todo 
     bpemb_encoder = BPEmb(lang="en", dim=100)
     feature_vector=[]
     for idx, word_info in word_dict.values():
#         print(word_info)
        word = word_info[1].strip()
        bool_feature = create_Boolean_feature(word,database)
        text_feature = create_text_feature(word,bpemb_encoder)
        feature =np.concatenate((bool_feature , num_feature[idx] , text_feature))
        feature_vector.append(feature)
     
     return feature_vector
    

def extract_bbox_info(img_path):
    hocr_data = pytesseract.image_to_pdf_or_hocr(img_path, extension='hocr')
    soup = BeautifulSoup(hocr_data)
    spans = soup.find_all('span', attrs={'class':'ocrx_word'})  # getting the attributes 
    word_bbox_list =[]  
    bbox_width=[]
    for span in spans:
        bbox_str=span['title']  # geting bounding box inforamtion 
        bbox_str =bbox_str.split(';')
        bbox =bbox_str[0].split()[1:]
    #     x,y,w,h= *bbox
    #     word_info_list.append([span['id'],span.string,[bbox[1].split()[-1],*bbox]])
        word_bbox_list.append([span['id'],span.string,
                               [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]
                              ])
        bbox_width.append((int(bbox[2])-int(bbox[0])))
#         word_info_list.sort(key = lambda sort_value: (sort_value[2][1],sort_value[2][0]))
#     word_info_list.sort(key = lambda sort_value: sort_value[2][1])
    
    return word_bbox_list, bbox_width


def remove_noise(bbox_width, word_list):
    mean=np.mean(bbox_width) # mean of the width of bounding box
    sd=np.std(bbox_width)  #stanered deviation of the width of the bounding box
    sd=round(sd) #Roinding to integer value and find the threshold value
#     print(mean,sd)
    def noise_removal_finction(box_info):
        box_width = box_info[2][2] - box_info[2][0]
#         print(box_width)
        if  (mean*2-sd) <=box_width < mean+sd:
            if box_info[1] !=' ' and box_info[1] !='-':
                return box_width
    
    filtered_word_list = filter(noise_removal_finction, word_list)
    return list(filtered_word_list)


def remove_noise_info(bbox_width, word_list):
    mean=np.mean(bbox_width) # mean of the width of bounding box
    sd=np.std(bbox_width)  #stanered deviation of the width of the bounding box
    sd=round(sd) #Roinding to integer value and find the threshold value
#     print(mean,sd)
    filtered_word_list=[]
    remove_word_list=[]
    for box_info in word_list:
        box_width = box_info[2][2] - box_info[2][0]
#         print(box_width)
        if  (mean*2-sd) <=box_width < mean+sd:
            if box_info[1] !=' ':
                filtered_word_list.append(box_info)
            else:
                remove_word_list.append(box_info)
        else:
            remove_word_list.append(box_info)
    
    return remove_word_list, filtered_word_list

def Line_Formation(word_list):
    word_list.sort(key =lambda sort_on_top : sort_on_top[2][1]) # sorting on top value 
    line_list =[]  
    word_list_len = len(word_list)
    temp=[]
    temp.append(word_list[0])   # intializing the value with pre-assumption 
    for i in range(1,word_list_len, 1):
        wa_top =word_list[i-1][2][1]
        wa_bottom = word_list[i-1][2][3]

        wb_top =word_list[i][2][1]
        wb_bottom = word_list[i][2][3]
        if wa_top <= wb_bottom and wa_bottom >= wb_top:
            temp.append(word_list[i])
        else:
            temp.sort(key=lambda sort_on_left_ : sort_on_left_[2][0]) # sorting on the left value
            line_list.append(temp)
            temp=[]
            temp.append(word_list[i])
    line_list.append(temp)
    return line_list

        
def get_image_data(folder_path,file_name=False):
    file_path = glob.glob(folder_path)
    img_list =[]
    file_name=[]
    for path in file_path:
        if path.split('.')[-1]=='pdf':
            print('pdf file skiped')
            continue
#        print(path)
        img = cv2.imread(path)
        img_list.append(img)
        file_name.append(path)
    if file_name:
        return img_list,file_name
    return img_list




def display_image(img):
    res = isinstance(img, str)
    if res:
        img = cv2.imread(img)
        
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', img)

    
def display_bounding_box(img, word_bbox_list,title="word bounding box"):
    for bbox in word_bbox_list:
        b = bbox[-1]
        img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

    # show annotated image and wait for keypress
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
 

def display_connected_graph(img,lookup,conn_type=None,isbox=False):
    
    for key in lookup.keys():    
        if isbox or conn_type==None: 
             b =  word_dict[key][1][2]
             img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
             if conn_type==None :
                 continue
        if conn_type=='right' or conn_type=='merge': 
           
            if lookup.get(key).get('right') == None:
                continue    
            right_value = lookup[key]['right'][0]
            right_source = word_dict[key][1][2]
            right_target= word_dict[right_value][1][2] 
            y1 =sorted([right_source[1],right_source[3],right_target[1]])[1]
            y2 =sorted([right_source[1],right_source[3],right_target[3]])[1]
            y=round((y1+y2)/2)
            start_point = (right_source[2],y)
            end_point = (right_target[0],y)
#            print(start_point,end_point)
            img = cv2.line(img, (start_point),(end_point), (0, 0, 255), 2)
            
        if conn_type=='bottom'or conn_type=='merge':
            if lookup.get(key).get('bottom') == None:
                continue
            bottom_value = lookup[key]['bottom'][0]
            bottom_source = word_dict[key][1][2]
            bottom_target= word_dict[bottom_value][1][2]
#            print('bottom_source {} target {}  point stored {}'.format(bottom_source, bottom_target,word_dict[key][1]))
            x1 =sorted([bottom_source[0],bottom_target[0],bottom_target[2]])[1]
            x2 =sorted([bottom_source[2],bottom_target[0],bottom_target[2]])[1]
            x=round((x1+x2)/2)
            start_point = (x,bottom_source[3])
            end_point = (x,bottom_target[1])
#            print(start_point,end_point)
            img = cv2.line(img, (start_point),(end_point), (0, 255, 255), 2)
        
            
    cv2.namedWindow('connected node Image', cv2.WINDOW_NORMAL)
    cv2.imshow('connected node Image', img)
    cv2.waitKey(0)
            
def display_graph(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    plt.show()


def find_right_NN(row,col,lines,page_width,lookup):
    min_dist =99999
    RD_R=99999
    col_size = len(lines[row])
    source =lines[row][col]
#    print('source : ', source)
    source_TR= source[2][1]   # assume the co-ordinates as (y,x) and bounding box has (y,x,h,w) ->(left, top, right ,bottom)
    source_BR = source[2][3]
    for j in range(col+1, col_size):
        target = lines[row][j]
        target_TL = target[2][1]
        target_BL= target[2][3]
        
        case1 =source_TR <= target_TL <= source_BR       # case 1: when target top overlap with source  (horizontally)
        case2 =source_TR <= target_BL<= source_BR # case 2: when target bottom overlap with source in right
        case3 =target_TL < source_TR < target_BL     # case 3: when source is under overlap with target
        if case1 or case2 or case3:
            # calculaing relative distance WRT soruce and targe 
            #RD_R = left(targe)-right(soruce)/page_width
            RD_R = (target[2][0] - source[2][2])/page_width
            if(min_dist > RD_R):   # tracking minimum distance 
                if not bool(lookup.get(target[0]).get('left'))   :  # if target is empty for source in lookup 
                    lookup[source[0]]['right']=[target[0],RD_R]
                    lookup[target[0]]['left']= [source[0],RD_R]
                    min_dist=RD_R
#                    print('target : ', target)
                else:
                    if lookup[target[0]]['left'][1] > RD_R:   # if target is Not empty for source then check left distance
                        remove_source = lookup[target[0]]['left'][0]
                        lookup[target[0]]['left']= [source[0],min_dist]
                        lookup[remove_source]['right']=None
                        min_dist=RD_R
#                        print('update target : ',target)
#                        print('update remove_source : ',remove_source)
                        
                
        if RD_R > min_dist: # tracking prev_dist for stoping the seraching of NN  
            break       
    
    return lookup


def find_bottom_NN(row,col,lines,page_hight,lookup):
    min_dist =99999
    RD_B=99999
    row_size = len(lines)
    source =lines[row][col]
#    print('source : ',source)
    source_BL = source[2][0]  # assume the co-ordinates as (y,x) and bounding box has (y,x,h,w) ->(left, top, right ,bottom)
    source_BR = source[2][2]
    for i in range(row+1,row_size):
        col_size = len(lines[i]) # finding the column of the line
        for j in range(0, col_size):       
            target = lines[i][j]           
            target_TL = target[2][0]
            target_TR = target[2][2]
            case1 =source_BL <= target_TR <= source_BR  # case 1. target right overlap with source in bottom (vertically)
            case2 =source_BL <= target_TL<=source_BR   # case 2. target left overlap with source  in bottom
            case3 =target_TL < source_BL < target_TR   #case 3 . source overlap under target  in bottom
            
            if case1 or case2 or case3:    
                # calculaing relative distance WRT soruce and targe 
                #RD_B = top(targe)-bottom(soruce)/page_hight
                RD_B = (target[2][1] - source[2][3])/page_hight               
                if(min_dist > RD_B):   # tracking minimum distance 
                    if not bool(lookup.get(target[0]).get('top')) :    # if source is empty than updated in lookup 
                        min_dist=RD_B
                        lookup[source[0]]['bottom']=[target[0],min_dist]
                        lookup[target[0]]['top']=[source[0],min_dist]
#                        print('target : ',target)
                        
                    else:
                        if(lookup[target[0]]['top'][1] > RD_B):   # source is updated and checking which one having less min_dist
                            min_dist=RD_B
                            remove_source = lookup[target[0]]['top'][0]
                            lookup[target[0]]['top']=[source[0],min_dist]
                            lookup[remove_source]['bottom']=None
#                            print('update target : ',target)
#                            print('update remove_source : ',remove_source)
                        
            if RD_B > min_dist : # tracking prev_dist for stoping the seraching of NN
                i= row_size + 1 # terminating the seraching of NN in bottom 
#                 print('breaking')
                break
                
    return lookup


def get_NN_lookup_table(lines,page_width,page_hight):
    
    lookup={}
    for row,line in  enumerate(lines):
        for col,word in  enumerate(line):
            lookup[word[0]]={}

    for line_idx, line in enumerate(lines):    
        for word_idx, word in enumerate(line): 
            lookup= find_right_NN(line_idx,word_idx,lines,page_width,lookup) 
            lookup= find_bottom_NN(line_idx,word_idx,lines,page_hight,lookup)
    return lookup

def get_word_dict(lines):    
    word_dict ={}
    contain_dict={}
    word_count=0
    for line in lines:
        for word in line:
            word_dict[word[0]]=[word_count,word]
            contain_dict[word[0]]=word[1]
            word_count+=1
    return word_dict,contain_dict


def create_graph_numeric_feature(word_dict,lookup):
    word_id_list = word_dict.keys()
    size = len(word_id_list)
    adj_mat = np.zeros((size,size))
    num_feature = np.zeros((size,4))
    
    for row, word_id in enumerate(word_id_list):
        word_NN = lookup[word_id]
        
        if bool(word_NN.get('left')):
            idx , weight = word_NN.get('left')
            col = word_dict[idx][0]
            adj_mat[row][col]= -weight
            num_feature[row][0]=-weight
        
        if bool(word_NN.get('top')):
            idx , weight = word_NN.get('top')
            col = word_dict[idx][0]
            adj_mat[row][col]= -weight
            num_feature[row][1]=-weight
        
        if bool(word_NN.get('right')):
            idx , weight = word_NN.get('right')
            col = word_dict[idx][0]
            adj_mat[row][col]= weight
            num_feature[row][2]=weight
            
        if bool(word_NN.get('bottom')):
            idx , weight = word_NN.get('bottom')
            col = word_dict[idx][0]
            adj_mat[row][col]= weight
            num_feature[row][3]=weight
            
    return adj_mat,num_feature

def create_label_feature(entity):
    entity_labels=['label-0', 'label-1', 'label-2', 'label-3', 'label-4', 'label-5', 'label-6', 'label-7', 
                   'label-8', 'label-9', 'label-10', 'label-11', 
                   'label-12', 'label-13', 'label-14', 'label-15', 'label-16', 'label-17', 'label-18', 'label-19']
    
    entity=entity.lower()
    label=np.zeros((len(entity_labels)))
    try:
        label[entity_labels.index(entity)]=1
    except ValueError:
        label[19]=1
    finally:
        return label

#--------------------------------model creation ---------------------------------------------------------
   
class CreateDataset(Dataset):
    def __init__(self, adj_mat_list, feature_vector_list, label_list=None):
        self.adj_mat =adj_mat_list
        self.feature=feature_vector_list
        self.label= label_list
    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        A = csr_matrix(self.adj_mat[idx])
        edge_index, edge_attr = convert.from_scipy_sparse_matrix(A)
        x=torch.tensor(self.feature[idx], dtype=torch.float)
        
        if self.label != None:
            y=torch.tensor(self.label[idx], dtype=torch.long)
            data =Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        else:
            data =Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        return data
        

class Net(torch.nn.Module):
    def __init__(self,features_size):
        super(Net, self).__init__()
        self.conv1 = ChebConv(features_size, 16,3)
        self.conv2 = ChebConv(16, 32,3)
        self.conv3 = ChebConv(32, 64,3)
        self.conv4 = ChebConv(64, 128,3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)    
    
def get_validation_loss(validloder, model, loss_fun, epoch,betch_size=2, train_on_gpu=False):
     #validation loss     
    valid_loss = 0.0
    valid_acc = 0.0

    # Don't need to keep track of gradients
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()
        # Validation loop
        for data in validloder:
            # Tensors to gpu
            if train_on_gpu:
                data= data.cuda()
            # Forward pass
            output = model(data)
#             print('output size {}:\n{} '.format(output.size(),output))
            # Validation loss
            loss = loss_fun(output, data.y.long())
#             pred = output.max(dim=1)[1]
            # Multiply average loss times the number of examples in batch
            valid_loss += loss.item() * betch_size

            # Calculate validation accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(data.y.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples
            valid_acc += accuracy.item() * betch_size

        # Calculate average losses
        valid_loss = valid_loss / len(validloder.dataset)

        # Calculate average accuracy
        valid_acc = valid_acc / len(validloder.dataset)
        
#         print(f'\nEpoch: {epoch} \tValidation Loss: {valid_loss:.4f} \tValidation Accuracy: {100 * valid_acc:.2f}')
        
    return valid_loss, valid_acc


def get_taining_loss(trainloader,model, optimizer,loss_fun,epoch,betch_size=2,train_on_gpu=False):
    # keep track of training and validation loss each epoch
    train_loss = 0.0
    train_acc = 0
    
    # Set to training mode
    model.train()
    
    # trainig loop 
    for i, data in enumerate(trainloader, 0):

        # Tensors to gpu
        if train_on_gpu:
            print('running on gpu')
            data = data.cuda()
            
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data)
#         print(type(output))
        loss = loss_fun(output, data.y.long())
        loss.backward()
        optimizer.step()

        # Track train loss by multiplying average loss by number of examples in batch
        train_loss += loss.item() * betch_size

        # Calculate accuracy by finding max log probability
        _, pred = torch.max(output, dim=1)
        correct_tensor = pred.eq(data.y.view_as(pred))
        # Need to convert correct tensor from int to float to average
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
       
        # Multiply average accuracy times the number of examples in batch
        train_acc += accuracy.item() * betch_size
        
        # Calculate average losses
        train_loss = train_loss / len(trainloader)
       
        # Print training and validation results
                
#         print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tTraining Accuracy: {100 * train_acc:.2f}')
    return train_loss, train_acc



def train_model(model,model_path,trainloder,validloder,No_epochs=2000,max_epochs_stop=50,print_every=100):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    Entropy_loss = nn.CrossEntropyLoss()
    
    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    history = []
    overall_start=timer()
    for epoch in range(No_epochs):
        start=timer()
        train_loss, train_acc = get_taining_loss(trainloder,model,optimizer,Entropy_loss,epoch)
        
        valid_loss, valid_acc = get_validation_loss(validloder,model,Entropy_loss,epoch)
        
        history.append([train_loss, valid_loss, train_acc, valid_acc])  # adding data for plot
        
        # Print training and validation results
        if (epoch + 1) % print_every == 0:
            print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \t\tValidation Loss: {valid_loss:.4f}')
#             print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\tValidation Accuracy: {100 * valid_acc:.2f}%')
            print(f'\t\t\t( {timer() - start:.2f} seconds elapsed in epoch {epoch})',end='\r')
        
        # Save the model if validation loss decreases
        if valid_loss < valid_loss_min:
            # Save model
            torch.save(model.state_dict(), model_path)
            # Track improvement
            epochs_no_improve = 0
            valid_loss_min = valid_loss
#            valid_best_acc = valid_acc
            best_epoch = epoch
        # Otherwise increment count of epochs with no improvement
        else:
            epochs_no_improve += 1    
            
        # Trigger early stopping
        if epochs_no_improve >= max_epochs_stop or epoch+1 >= No_epochs:
            if( epoch+1 >= No_epochs):
                print(f'\n-----------------------------------Traning completed----------------------------------')
                print(f'\nCompleted Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
            else:
                print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
                
            total_time = timer() - overall_start
            print(f'\t({total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch)')

            # Load the best state dict
            model.load_state_dict(torch.load(model_path))
            # Attach the optimizer
            model.optimizer = optimizer

            # Format history
            history = pd.DataFrame(history,columns=['train_loss', 'valid_loss', 'train_acc','valid_acc'])
            return model, history


def plot_loss(history):
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    
def plot_accuracy(history):
    plt.figure(figsize=(8, 6))
    for c in ['train_acc', 'valid_acc']:
        plt.plot(100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')


def image_feature_processing(img_data=None,folder_path=None):
    if img_data==None:
        img_data,file_name = get_image_data(folder_path,True)
    
    adj_mat_list=[]
    feature_vector_list=[]
    word_info_list=[]
    for idx,img in enumerate(img_data): 
        if folder_path !=None:
            print('processing file (id {}) : {}'.format(idx,file_name[idx]))
            
        print('processing file id : {}'.format(idx))
        word_bbox_list, bbox_width =  extract_bbox_info(img)
        word_list = remove_noise(bbox_width,word_bbox_list)
        lines = Line_Formation(word_list)
        word_dict , contain_dict= get_word_dict(lines)
        word_info_list.append([word_list,word_dict,contain_dict])
        width, hight ,_ = img.shape
        lookup= get_NN_lookup_table(lines,width,hight)
        adj_mat, num_feature = create_graph_numeric_feature(word_dict,lookup)
        feature_vector = create_feature_vector(word_dict,num_feature)
        adj_mat_list.append(adj_mat)
        feature_vector_list.append(feature_vector)
        
    return adj_mat_list,feature_vector_list,word_info_list


def print_ocr_string(img):  
    res = isinstance(img, str)
    if res:
        print('reading image ..\n' )
        img = cv2.imread(img)  
    text = pytesseract.image_to_string(img)
#     print(text)
    return text

```


```python
#--------------------------------main ---------------------------------------------
    
    
pincode_path="C:/Users/Sanjeev/Documents/Python Scripts/cheat Sheet/Pincode_30052019.csv"
# pincode_path= "C:/Users/Sanjeev/Documents/Accenture office/Pincode_30052019.csv"
# country_path="C:/Users/Sanjeev/Documents/Accenture office/countries.csv"
country_path="C:/Users/Sanjeev/Documents/Accenture office/countries.txt"
database_path ="C:/Users/Sanjeev/Documents/Accenture office/database"
img_path="C:/Users/Sanjeev/Documents/Accenture office/invoice3.png"
tesseract_path ='C:/python_lib/ocr/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = 'C:/python_lib/ocr/tesseract.exe'
model_path = "C:/Users/Sanjeev/Documents/Accenture office/invoice_model"
folder_path = "C:/Users/Sanjeev/Documents/Accenture office/invoice data/*"

# folder_path = "C:/Users/Sanjeev/Documents/Python Scripts/Dataset/output/*.png"
csv_path ='C:/Users/Sanjeev/Documents/Python Scripts/Dataset/csv/'
data_path = "C:/Users/Sanjeev/Documents/Python Scripts/Dataset/df_word_list"
```


```python
# reading the image file from the folder and stored as numpy array
img_data, file_name = get_image_data(folder_path,True)
print(len(img_data))
#1,2,3,5  (issue in noise removal-> 4)
# del img_data[4]
# del file_name[4]
print(len(img_data))
```

    pdf file skiped
    pdf file skiped
    6
    6
    


```python
# pre-processing the image data 
# 1. feed the image to OCR and get HOCR file which contain bounding box infomation
# 2. Create numeric, boolean, and word features 
# 3. Create Graph 
# 4. return adjacency metric A, feature vector 317 dim, and word infoamtion 
adj_mat_list,feature_vector_list, word_info_list = image_feature_processing(img_data=img_data[0:3])    # pre-processing and graph Creation
```

    processing file id : 0
    processing file id : 1
    processing file id : 2
    


```python
np.array(feature_vector_list[0]).shape

```


```python
# creating label for the each node of the graph (dummy label for testing Neural Network pipeline)
label_list=[]
seed=1234567833
np.random.seed(seed)
for i in range(len(adj_mat_list)):
    label = np.random.randint(0,10,adj_mat_list[i].shape[0])
    label_list.append(label)

```


```python
# creating dataset and dataloader for pytorch pipeline
dataset =CreateDataset(adj_mat_list,feature_vector_list,label_list)

betch_size=2
train_dataloder = DataLoader(dataset,betch_size)
torch.manual_seed(seed)
model=Net(317)
model , history = train_model(model,model_path,train_dataloder,train_dataloder,10,5,10) # train the Neural Network 

```

    
    Epoch: 9 	Training Loss: 40.6291 		Validation Loss: 43.2771
    			( 0.11 seconds elapsed in epoch 9)
    -----------------------------------Traning completed----------------------------------
    
    Completed Total epochs: 9. Best epoch: 9 with loss: 43.28 and acc: 21.75%
    	(1.31 total seconds elapsed. 0.13 seconds per epoch)
    


```python
# ploting training and validation loss 
plot_loss(history)

# ploting training and validation accuracy 
# plot_accuracy(history)
```


```python
# testing he model after training 

## yet to impement
```


```python
# visualization of image, word detected by ocr, words after removal of noise, and Graph generated.

idx=1
img=cv2.imread(file_name[idx])
display_image(img)

word_bbox_list, bbox_width =  extract_bbox_info(img)
display_bounding_box(img, word_bbox_list,'orginal words')

# print_ocr_string(img)

word_list = remove_noise(bbox_width,word_bbox_list)
img=cv2.imread(file_name[idx])
display_bounding_box(img, word_list,'word after removal')

lines = Line_Formation(word_list)
word_dict , contain_dict= get_word_dict(lines)
width, hight ,_ = img.shape
lookup= get_NN_lookup_table(lines,width,hight)
adj_mat, num_feature = create_graph_numeric_feature(word_dict,lookup)

img = cv2.imread(file_name[idx])
display_connected_graph(img, lookup, 'merge',True)

cv2.waitKey(0)
# feature_vector = create_feature_vector(word_dict,num_feature)
```


```python
img = cv2.imread(file_name[idx])
display_connected_graph(img, lookup, 'merge',True)
```


```python
lines

```


```python
No_feature=317
def get_prediction(model,img_list=None,img_path=None):
#     model.eval()
    if img_list != None:
        img_data= img_list
    if img_path !=None:
        img_data=img_path
    adj_mat_list,feature_vector_list, word_info_list = image_feature_processing(img_data=img_data)
    
    betch_size = len(adj_mat_list)   # no of data for testing 
    
    if betch_size==1:
        print('betch_size : ',betch_size)
        test_x=torch.tensor(feature_vector_list[0], dtype=torch.float) 
        test_edge_index, test_edge_attr = convert.from_scipy_sparse_matrix(csr_matrix(adj_mat_list[0]))
        test_data=Data(x=test_x,edge_index=test_edge_index,edge_attr=test_edge_attr)
    else:
        print('betch_size : ',betch_size)
        test_data =CreateDataset(adj_mat_list, feature_vector_list)
        test_data = DataLoader(test_data,betch_size)
        test_data = next(iter(test_data))

    pred_out = model(test_data) 
    print(pred_out.size())
    _ ,pred_out = torch.max(pred_out, dim=1)
    
    return pred_out
            
    
    
```


```python
test_model.load_state_dict(torch.load(model_path))
test_model.eval()
# pred =get_prediction(test_model,img_data)
# print(pred.size())
# correct_tensor = pred.eq(test_y)
# print(sum(correct_tensor))
# print(correct_tensor)
```


```python
p_o=pred.numpy()
p_o[0:5]
```


```python
sum=0
for label in label_list:
    print(len(label))
    sum+= len(label)
sum
```


```python
test_idx=1

test_x=torch.tensor(feature_vector_list[test_idx], dtype=torch.float)
test_y=torch.tensor(label_list[test_idx], dtype=torch.long)
# A = csr_matrix(adj_mat_list[test_idx])
test_edge_index, test_edge_attr = convert.from_scipy_sparse_matrix(csr_matrix(adj_mat_list[test_idx]))
test_data=Data(x=test_x,edge_index=test_edge_index,edge_attr=test_edge_attr)
test_model = Net(317)
test_model.load_state_dict(torch.load(model_path))
test_model.eval()

```


```python
pred_out= test_model(test_data)
```


```python
a, pred = torch.max(pred_out, dim=1)
correct_tensor = pred.eq(test_y)
# correct_tensor.numpy()
```


```python
import spacy

nlp = spacy.load("en_core_web_sm")

text =f'sanjeev has good heart person and he is nice.His DOB is 15/05/1992.'
# text =text.split(' ')[3]
doc=nlp(text)
# print(doc)

# for token in doc:
#     print('{} -> {}'.format(token.text, token.pos_))

for token in doc.ents:
    print('{} -> {}'.format(token.text, token.label_))
```


```python
#this function takes input word and out p
# ORG -> [1,0,0,0]
# Date -> [0,1,0,0]
# GPE ->  [0,0,1,0]
# MONEY/CARDINAL->[0,0,0,1]
#except this -> [0,0,0,0]

def get_ner_tag(ner_tag):
    if len(ner_tag) is not 0:
        ner_tag=ner_tag[0].label_
        print(ner_tag)
        if ner_tag=='ORG':
            return [1,0,0,0]
        elif ner_tag=='DATE':
            return [0,1,0,0]
        elif ner_tag=='GPE':
            return [0,0,1,0]
        elif ner_tag=='MONEY' or ner_tag=='CARDINAL':
            return [0,0,0,1]
        else:
            return [0,0,0,0]
    else:
         return [0,0,0,0]
    
    

#this function takes token/word as input and give 2-d output( 10: -> noun/pronoun,01-> Number, 00-> others )
# [1,0,0]

def get_pos_tag(pos_tags):
    
    if pos_tags =='NOUN' or pos_tags=='PRON':
        return [1,0]
    elif pos_tags =='NUM':
        return [0,1]
    else:
        return [0,0]
        
    

def create_tag_features(word_list):
    nlp = spacy.load("en_core_web_sm")
    pos_tag_list = []
    ner_tag_list=[]
    for word in word_list:
        tag =nlp(word)
        pos_tag_list.append(get_pos_tag(tag[0].pos_))
        ner_tag_list.append(get_ner_tag(tag.ents))
    return pos_tag_list , ner_tag_list
    
    
```


```python
print(text)
pos, ner =create_tag_features(text.split(' '))
```


```python
print(pos,ner)

```


```python
idx=0
img=cv2.imread(file_name[idx])
# display_image(img)

word_bbox_list, bbox_width =  extract_bbox_info(img)
# word_bbox_list =noise_remove(word_bbox_list)
display_bounding_box(img, word_bbox_list,'orginal words')
cv2.waitKey(0)
```


```python
def noise_remove(word_list,skip_symbol=None):
    filter_list=[]
    for w_list in word_list:
        word= w_list[1]
        if len(word)==1: # checking if word length is 1 it could be noise
            if skip_symbol and word in skip_symbol:   # if word is in our desired symbol ex:- $,₹, etc 
                filter_list.append(w_list)
            elif is_alphaNumeric(word):   # consider the case John D :- d has len one but it is alphabet.
                filter_list.append(w_list)
        else:
            filter_list.append(w_list)
    return filter_list
            
            

```


```python

```


```python
W_list = [ ['word_1_26', ' ', [521, 108, 542, 119]],
 ['word_1_27', ' ', [42, 232, 714, 236]],
 ['word_1_28', '#', [48, 244, 56, 253]],
 ['word_1_39', 'x10', [415, 272, 439, 282]],    
['word_1_37', '1', [49, 272, 53, 282]],
        ['word_1_43', '=', [75, 294, 79, 295]],
        ['word_1_83', '-', [75, 448, 79, 449]],
        ['word_1_86', ' ', [115, 413, 124, 420]],
 ['word_1_87', ' ', [42, 461, 714, 465]],
 ['word_1_88', ' ', [407, 481, 481, 504]],
 ['word_1_89', ' ', [485, 513, 490, 527]],
       ]


```


```python

lines=W_list
sum([len(line) for line in lines])
```


```python
noise_remove(W_list)
```


```python
database = LoadData(database_path) #todo 
```


```python

```


```python
if lookup.get(key).get('right') == None:
                continue    
            right_value = lookup[key]['right'][0]
            right_source = word_dict[key][1][2]
            right_target= word_dict[right_value][1][2] 
            y1 =sorted([right_source[1],right_source[3],right_target[1]])[1]
            y2 =sorted([right_source[1],right_source[3],right_target[3]])[1]
            y=round((y1+y2)/2)
            start_point = (right_source[2],y)
            end_point = (right_target[0],y)
#            print(start_point,end_point)
            img = cv2.line(img, (start_point),(end_point), (0, 0, 255), 2)
            
        
```


```python
lines[0]
```




    [['word_1_4', '"Romashka"', [49, 44, 120, 54]],
     ['word_1_5', 'Ltd.', [125, 44, 145, 54]],
     ['word_1_1', 'Invoice', [411, 34, 460, 45]],
     ['word_1_2', 'ID:', [465, 34, 483, 45]],
     ['word_1_3', 'INV/20111209-22', [567, 34, 679, 45]]]




```python
def row_merge(lines, row_threshold=10):
    row_bbox_list=[]
    merge_info=[]
    for line in lines:
        curr_bbox = line[0][2]
        print(line)
        print('curr_bbox ',curr_bbox)
        temp=[]
        for idx, word_bbox in enumerate(line):
            next_bbox=word_bbox[-1]
            print('next_bbox ',next_bbox)
            dist = np.sqrt((curr_bbox[2] - next_bbox[0])**2)    # calculating distance having y point common(mid point in overlaping case)
            print('dist : ', dist)
            if dist <= row_threshold:
                curr_bbox[2]= max(curr_bbox[2],next_bbox[2])
                curr_bbox[3]= max(curr_bbox[3],next_bbox[3])
                temp.append(word_bbox)
                print('curr_bbox after merge : ',curr_bbox)
            else:
                row_bbox_list.append(curr_bbox)
                curr_bbox=word_bbox[-1]
                print('curr_bbox after adding : ',curr_bbox)
                merge_info.append(temp)
        print('\n---------------------------line--------------------------------\n')
    return row_bbox_list,merge_info
```


```python

     
    
    
    
def col_clustring(merge_lines, col_threshold=4):
    col_bbox_list =[]
#     merge_info=[] #  for storing the merge info
    
#     for i in range(len(merge_lines)-1):
    curr_bbox_id =0
    for idx, top_row in enumerate(merge_lines):
        if idx+1 >= len(merge_lines):
            break
        bottom_row= merge_lines[idx+1]
        for top_bbox in top_row:
            for  bottom_bbox in bottom_row:
                 if check_overlaping(top_bbox, bottom_bbox):
                        dist = np.sqrt((top_bbox[3] - bottom_bbox[1]**2))
                        if dist <=col_threshold:
                        top_bbox[0]= max(top_bbox[0],bottom_bbox[0])
                        top_bbox[1]= min(top_bbox[1],bottom_bbox[1])
                        top_bbox[2]= max(top_bbox[2],bottom_bbox[2])
                        top_bbox[3]= max(top_bbox[3],bottom_bbox[3])
                        bottom_bbox=top_bbox
                    
                    elif(top_bbox[2] < bottom_bbox[0]):
                    break
                    
#                 else:
#                     col_bbox_list.append(top_bbox)
        
        col_bbox_list.append(top_bbox)


def
                            
                
                
        curr_bbox = merge_row[curr_bbox_id]
        next_bbox_id=0
#         for j in range(len(merge_lines[i+1])):
        total_word = len(merge_lines[idx+1])
        while next_bbox_id < total_word:
            next_bbox = merge_row[next_bbox_id]
            
            if check_overlaping(curr_bbox,next_bbox):
                dist = np.sqrt((curr_bbox[3] - next_bbox[1]**2))
                if dist <=col_threshold:
                    
                    curr_bbox[0]= max(curr_bbox[0],next_bbox[0])
                    curr_bbox[1]= min(curr_bbox[1],next_bbox[1])
                    curr_bbox[2]= max(curr_bbox[2],next_bbox[2])
                    curr_bbox[3]= max(curr_bbox[3],next_bbox[3])
                    
                else:
                    
                    col_bbox_list.append(curr_bbox)
                    curr_bbox_id=
                    
                next_bbox_id +=1

                
                
```


      File "<tokenize>", line 22
        elif(top_bbox[2] < bottom_bbox[0]):
        ^
    IndentationError: unindent does not match any outer indentation level
    



```python
def row_clustring(lines_list, row_threshold=4):
    row_bbox_list =[]
    test_list=[]
    merge_info=[] #  for storing the merge info
    for idx,line in enumerate(lines_list):
        temp_test_list=[]
        curr_bbox_id=0
        next_bbox_id=1
        prev_id=-1
        total_word=len(line) 
        print(line)
        print('total word = ',total_word)
        temp_merge=[]
        info=[]
#         temp_merge.append(line[curr_bbox_id])
        curr_bbox=line[curr_bbox_id][2]
        print('curr_bbox : ',curr_bbox)
        
        while next_bbox_id < total_word:
            next_bbox=line[next_bbox_id][2]
            print('next_bbox : ',line[next_bbox_id])
            dist = np.sqrt((curr_bbox[2] - next_bbox[0])**2)    # calculating distance having y point common(mid point in overlaping case)
#             dist = curr_bbox[2] - next_bbox[0] 
            print('dist ',dist)
            if dist <= row_threshold:    # merging two bbox
#                 print('min ----------------------dist-: ',dist)
                curr_bbox[2]= max(curr_bbox[2],next_bbox[2])
                curr_bbox[3]= max(curr_bbox[3],next_bbox[3])
                print('curr_bbox (merging)',curr_bbox)
                if prev_id != curr_bbox_id:
                    info.append(line[curr_bbox_id])
                    prev_id=curr_bbox_id
                info.append(line[next_bbox_id])      
            else:
                row_bbox_list.append(curr_bbox)
                print('row_bbox_list (inserting)',curr_bbox)
                temp_test_list.append(curr_bbox)  # tesing 
                
                temp_merge.append(info)
                curr_bbox_id= next_bbox_id
                curr_bbox=next_bbox
                print('curr_bbox (adding)',curr_bbox)
                info=[]
            next_bbox_id +=1
            print('next_bbox_id ', next_bbox_id)
        merge_info.append(temp_merge)
        if dist > row_threshold:
            row_bbox_list.append(line[next_bbox_id-1][2])
            print('row_bbox_list (inserting) after last word',line[next_bbox_id-1])
            temp_test_list.append(line[next_bbox_id-1][2])
        else:
            row_bbox_list.append(curr_bbox)
            print('row_bbox_list (inserting) after merge',curr_bbox)
            temp_test_list.append(curr_bbox)
        test_list.append(temp_test_list)
        print('---------------------------line {}-----------------------------'.format(idx+1))
    return row_bbox_list,merge_info,test_list
                
                
            
            
```


```python
def check_overlaping(source,target):
    source_BL = source[0]  # assume the co-ordinates as (y,x) and bounding box has (y,x,h,w) ->(left, top, right ,bottom)
    source_BR = source[2]           
    target_TL = target[0]
    target_TR = target[2]
    
    
    case1 =source_BL <= target_TR <= source_BR  # case 1. target right overlap with source in bottom (vertically)
    case2 =source_BL <= target_TL<=source_BR   # case 2. target left overlap with source  in bottom
    case3 =target_TL < source_BL < target_TR   #case 3 . source overlap under target  in bottom

    if case1 or case2 or case3: 
        return True
    else:
        return False
# we always check from top row to next bottom row for merging two boxes.             
def col_clestring(row_merge_list, col_threshold):
    print('enter for calculaton ')
    merge_list =[]
    merge_info=[]
    top_row= row_merge_list[0]
    for idx in range(1,len(row_merge_list)):
        print('loop ', idx)
        bottom_row = row_merge_list[idx]
        top_row_len, bottom_row_len = len(top_row), len(bottom_row)
        i, j,  candidate_row=0,0,[]
#         print('top_row ', top_row)
#         print('bottom_row ', bottom_row)
#         print('len of rows top ={} ,botttom {} , i= {}, j= {}'.format(top_row_len,bottom_row_len, i, j))
        merge_flag=False
        while i < top_row_len:
            while j < bottom_row_len:
                if (i>= top_row_len):
                    print('breaking top row loop')
                    break
                top_bbox = top_row[i]
                bottom_bbox= bottom_row[j]
                
                print('top_row ', top_row)
                print('bottom_row ', bottom_row)
                print('len of rows top ={} ,botttom {} , i= {}, j= {}'.format(top_row_len,bottom_row_len, i, j))
                print('top_bbox = {} \nbottom_bbox = {}'.format(top_bbox,bottom_bbox) )
                if check_overlaping(top_bbox,bottom_bbox):    # when top_bbox and bottom_bbox overlap each other 
                    print('overlaping  ')
                    dist = np.sqrt(( bottom_bbox[1] - top_bbox[3])**2)
                    print('dist : ', dist)
                    if dist <=col_threshold:
                        print('distance statisfy : ',j)
                        top_bbox[0]= min(top_bbox[0],bottom_bbox[0])
                        top_bbox[1]= min(top_bbox[1],bottom_bbox[1])
                        top_bbox[2]= max(top_bbox[2],bottom_bbox[2])
                        top_bbox[3]= max(top_bbox[3],bottom_bbox[3])
                        j +=1
                        merge_flag=True
                        
                        if j >= bottom_row_len:  # handling last box box when it merge 
                            candidate_row.append(top_bbox)
                            print('adding *last* candidate row (top box)  ', top_bbox)
                        
                    else:
                        print('distance not statisfy: ', j)
                        candidate_row.append(bottom_bbox)
                        print('adding to candidate row (bottom_bbox) ', bottom_bbox)
                        j +=1
                else:
                    print('Not overlaping  ')
                    if(top_bbox[2] < bottom_bbox[0]):    # bottom_bbox away(right most) from top_bbox 
                        if merge_flag:
                            candidate_row.append(top_bbox)
                            print('(right most) adding to candidate row (top box) ', top_bbox)
                        merge_list.append(top_bbox)
                        print('(right most)adding to mergeList  box ', top_bbox)
                        i +=1
                    else:
                        candidate_row.append(bottom_bbox)
                        print('(left most) adding to candidate row (bottom_bbox) ', bottom_bbox)
                        j += 1
                    
                print('value i= {} and j ={} '.format(i,j))
                print('merge_list : ', merge_list)
                print('candidate_row :',candidate_row)
                print('\n--exsiting the while 1 loop--\n') 
            
            while(j < bottom_row_len): # case when top row empty and bottom row has word 
                print('bottom bbox is not empty : ', j)
                candidate_row.append(bottom_row[j])
                j +=1
                print('value i= {} and j ={} '.format(i,j))
            while(i < top_row_len): # case when bottom row empty and top row has word 
                print('top box  is not empty : ', i)
                merge_list.append(top_row[i])
                i +=1
                print('value i= {} and j ={} '.format(i,j))
        print('exsiting the while 2 loop')
        print('merge_list : ', merge_list)
        print('candidate_row :',candidate_row)
        top_row = candidate_row
        print('candidate_row ', candidate_row)
        print('\n---------------------------------------------------------------\n')
#     merge_list.append(top_row)
    return  merge_list+top_row
            
```


```python
d = col_clestring(c, 10)

```

    enter for calculaton 
    loop  1
    top_row  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45]]
    bottom_row  [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    len of rows top =3 ,botttom 3 , i= 0, j= 0
    top_bbox = [49, 44, 368, 73] 
    bottom_bbox = [49, 60, 368, 73]
    overlaping  
    dist :  13.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [49, 60, 368, 73]
    value i= 0 and j =1 
    merge_list :  []
    candidate_row : [[49, 60, 368, 73]]
    
    --exsiting the while 1 loop--
    
    top_row  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45]]
    bottom_row  [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    len of rows top =3 ,botttom 3 , i= 0, j= 1
    top_bbox = [49, 44, 368, 73] 
    bottom_bbox = [411, 59, 498, 70]
    Not overlaping  
    (right most)adding to mergeList  box  [49, 44, 368, 73]
    value i= 1 and j =1 
    merge_list :  [[49, 44, 368, 73]]
    candidate_row : [[49, 60, 368, 73]]
    
    --exsiting the while 1 loop--
    
    top_row  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45]]
    bottom_row  [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    len of rows top =3 ,botttom 3 , i= 1, j= 1
    top_bbox = [411, 34, 483, 45] 
    bottom_bbox = [411, 59, 498, 70]
    overlaping  
    dist :  14.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [411, 59, 498, 70]
    value i= 1 and j =2 
    merge_list :  [[49, 44, 368, 73]]
    candidate_row : [[49, 60, 368, 73], [411, 59, 498, 70]]
    
    --exsiting the while 1 loop--
    
    top_row  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45]]
    bottom_row  [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    len of rows top =3 ,botttom 3 , i= 1, j= 2
    top_bbox = [411, 34, 483, 45] 
    bottom_bbox = [567, 58, 635, 69]
    Not overlaping  
    (right most)adding to mergeList  box  [411, 34, 483, 45]
    value i= 2 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45]]
    candidate_row : [[49, 60, 368, 73], [411, 59, 498, 70]]
    
    --exsiting the while 1 loop--
    
    top_row  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45]]
    bottom_row  [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    len of rows top =3 ,botttom 3 , i= 2, j= 2
    top_bbox = [567, 34, 679, 45] 
    bottom_bbox = [567, 58, 635, 69]
    overlaping  
    dist :  13.0
    distance not statisfy:  2
    adding to candidate row (bottom_bbox)  [567, 58, 635, 69]
    value i= 2 and j =3 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45]]
    candidate_row : [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  2
    value i= 3 and j =3 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45]]
    candidate_row : [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    candidate_row  [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    
    ---------------------------------------------------------------
    
    loop  2
    top_row  [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    bottom_row  [[411, 83, 476, 94], [567, 83, 638, 94]]
    len of rows top =3 ,botttom 2 , i= 0, j= 0
    top_bbox = [49, 60, 368, 73] 
    bottom_bbox = [411, 83, 476, 94]
    Not overlaping  
    (right most)adding to mergeList  box  [49, 60, 368, 73]
    value i= 1 and j =0 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    bottom_row  [[411, 83, 476, 94], [567, 83, 638, 94]]
    len of rows top =3 ,botttom 2 , i= 1, j= 0
    top_bbox = [411, 59, 498, 70] 
    bottom_bbox = [411, 83, 476, 94]
    overlaping  
    dist :  13.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [411, 83, 476, 94]
    value i= 1 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73]]
    candidate_row : [[411, 83, 476, 94]]
    
    --exsiting the while 1 loop--
    
    top_row  [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    bottom_row  [[411, 83, 476, 94], [567, 83, 638, 94]]
    len of rows top =3 ,botttom 2 , i= 1, j= 1
    top_bbox = [411, 59, 498, 70] 
    bottom_bbox = [567, 83, 638, 94]
    Not overlaping  
    (right most)adding to mergeList  box  [411, 59, 498, 70]
    value i= 2 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70]]
    candidate_row : [[411, 83, 476, 94]]
    
    --exsiting the while 1 loop--
    
    top_row  [[49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    bottom_row  [[411, 83, 476, 94], [567, 83, 638, 94]]
    len of rows top =3 ,botttom 2 , i= 2, j= 1
    top_bbox = [567, 58, 635, 69] 
    bottom_bbox = [567, 83, 638, 94]
    overlaping  
    dist :  14.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [567, 83, 638, 94]
    value i= 2 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70]]
    candidate_row : [[411, 83, 476, 94], [567, 83, 638, 94]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  2
    value i= 3 and j =2 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    candidate_row : [[411, 83, 476, 94], [567, 83, 638, 94]]
    candidate_row  [[411, 83, 476, 94], [567, 83, 638, 94]]
    
    ---------------------------------------------------------------
    
    loop  3
    top_row  [[411, 83, 476, 94], [567, 83, 638, 94]]
    bottom_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    len of rows top =2 ,botttom 2 , i= 0, j= 0
    top_bbox = [411, 83, 476, 94] 
    bottom_bbox = [411, 108, 520, 119]
    overlaping  
    dist :  14.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [411, 108, 520, 119]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69]]
    candidate_row : [[411, 108, 520, 119]]
    
    --exsiting the while 1 loop--
    
    top_row  [[411, 83, 476, 94], [567, 83, 638, 94]]
    bottom_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    len of rows top =2 ,botttom 2 , i= 0, j= 1
    top_bbox = [411, 83, 476, 94] 
    bottom_bbox = [566, 107, 687, 118]
    Not overlaping  
    (right most)adding to mergeList  box  [411, 83, 476, 94]
    value i= 1 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94]]
    candidate_row : [[411, 108, 520, 119]]
    
    --exsiting the while 1 loop--
    
    top_row  [[411, 83, 476, 94], [567, 83, 638, 94]]
    bottom_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    len of rows top =2 ,botttom 2 , i= 1, j= 1
    top_bbox = [567, 83, 638, 94] 
    bottom_bbox = [566, 107, 687, 118]
    overlaping  
    dist :  13.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [566, 107, 687, 118]
    value i= 1 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94]]
    candidate_row : [[411, 108, 520, 119], [566, 107, 687, 118]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =2 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94]]
    candidate_row : [[411, 108, 520, 119], [566, 107, 687, 118]]
    candidate_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    
    ---------------------------------------------------------------
    
    loop  4
    top_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    bottom_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    len of rows top =2 ,botttom 6 , i= 0, j= 0
    top_bbox = [411, 108, 520, 119] 
    bottom_bbox = [48, 244, 56, 253]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [48, 244, 56, 253]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94]]
    candidate_row : [[48, 244, 56, 253]]
    
    --exsiting the while 1 loop--
    
    top_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    bottom_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    len of rows top =2 ,botttom 6 , i= 0, j= 1
    top_bbox = [411, 108, 520, 119] 
    bottom_bbox = [76, 243, 147, 256]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [76, 243, 147, 256]
    value i= 0 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94]]
    candidate_row : [[48, 244, 56, 253], [76, 243, 147, 256]]
    
    --exsiting the while 1 loop--
    
    top_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    bottom_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    len of rows top =2 ,botttom 6 , i= 0, j= 2
    top_bbox = [411, 108, 520, 119] 
    bottom_bbox = [417, 243, 438, 256]
    overlaping  
    dist :  124.0
    distance not statisfy:  2
    adding to candidate row (bottom_bbox)  [417, 243, 438, 256]
    value i= 0 and j =3 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94]]
    candidate_row : [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256]]
    
    --exsiting the while 1 loop--
    
    top_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    bottom_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    len of rows top =2 ,botttom 6 , i= 0, j= 3
    top_bbox = [411, 108, 520, 119] 
    bottom_bbox = [467, 243, 498, 253]
    overlaping  
    dist :  124.0
    distance not statisfy:  3
    adding to candidate row (bottom_bbox)  [467, 243, 498, 253]
    value i= 0 and j =4 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94]]
    candidate_row : [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253]]
    
    --exsiting the while 1 loop--
    
    top_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    bottom_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    len of rows top =2 ,botttom 6 , i= 0, j= 4
    top_bbox = [411, 108, 520, 119] 
    bottom_bbox = [587, 234, 620, 282]
    Not overlaping  
    (right most)adding to mergeList  box  [411, 108, 520, 119]
    value i= 1 and j =4 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119]]
    candidate_row : [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253]]
    
    --exsiting the while 1 loop--
    
    top_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    bottom_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    len of rows top =2 ,botttom 6 , i= 1, j= 4
    top_bbox = [566, 107, 687, 118] 
    bottom_bbox = [587, 234, 620, 282]
    overlaping  
    dist :  116.0
    distance not statisfy:  4
    adding to candidate row (bottom_bbox)  [587, 234, 620, 282]
    value i= 1 and j =5 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119]]
    candidate_row : [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282]]
    
    --exsiting the while 1 loop--
    
    top_row  [[411, 108, 520, 119], [566, 107, 687, 118]]
    bottom_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    len of rows top =2 ,botttom 6 , i= 1, j= 5
    top_bbox = [566, 107, 687, 118] 
    bottom_bbox = [676, 234, 709, 282]
    overlaping  
    dist :  116.0
    distance not statisfy:  5
    adding to candidate row (bottom_bbox)  [676, 234, 709, 282]
    value i= 1 and j =6 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119]]
    candidate_row : [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =6 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118]]
    candidate_row : [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    candidate_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    
    ---------------------------------------------------------------
    
    loop  5
    top_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 272, 619, 282], [676, 272, 708, 282]]
    len of rows top =6 ,botttom 5 , i= 0, j= 0
    top_bbox = [48, 244, 56, 253] 
    bottom_bbox = [83, 272, 255, 298]
    Not overlaping  
    (right most)adding to mergeList  box  [48, 244, 56, 253]
    value i= 1 and j =0 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 272, 619, 282], [676, 272, 708, 282]]
    len of rows top =6 ,botttom 5 , i= 1, j= 0
    top_bbox = [76, 243, 147, 256] 
    bottom_bbox = [83, 272, 255, 298]
    overlaping  
    dist :  16.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [83, 272, 255, 298]
    value i= 1 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253]]
    candidate_row : [[83, 272, 255, 298]]
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 272, 619, 282], [676, 272, 708, 282]]
    len of rows top =6 ,botttom 5 , i= 1, j= 1
    top_bbox = [76, 243, 147, 256] 
    bottom_bbox = [415, 272, 439, 282]
    Not overlaping  
    (right most)adding to mergeList  box  [76, 243, 147, 256]
    value i= 2 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256]]
    candidate_row : [[83, 272, 255, 298]]
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 272, 619, 282], [676, 272, 708, 282]]
    len of rows top =6 ,botttom 5 , i= 2, j= 1
    top_bbox = [417, 243, 438, 256] 
    bottom_bbox = [415, 272, 439, 282]
    overlaping  
    dist :  16.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [415, 272, 439, 282]
    value i= 2 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256]]
    candidate_row : [[83, 272, 255, 298], [415, 272, 439, 282]]
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 272, 619, 282], [676, 272, 708, 282]]
    len of rows top =6 ,botttom 5 , i= 2, j= 2
    top_bbox = [417, 243, 438, 256] 
    bottom_bbox = [464, 272, 496, 282]
    Not overlaping  
    (right most)adding to mergeList  box  [417, 243, 438, 256]
    value i= 3 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256]]
    candidate_row : [[83, 272, 255, 298], [415, 272, 439, 282]]
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 272, 619, 282], [676, 272, 708, 282]]
    len of rows top =6 ,botttom 5 , i= 3, j= 2
    top_bbox = [467, 243, 498, 253] 
    bottom_bbox = [464, 272, 496, 282]
    overlaping  
    dist :  19.0
    distance not statisfy:  2
    adding to candidate row (bottom_bbox)  [464, 272, 496, 282]
    value i= 3 and j =3 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256]]
    candidate_row : [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282]]
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 272, 619, 282], [676, 272, 708, 282]]
    len of rows top =6 ,botttom 5 , i= 3, j= 3
    top_bbox = [467, 243, 498, 253] 
    bottom_bbox = [587, 272, 619, 282]
    Not overlaping  
    (right most)adding to mergeList  box  [467, 243, 498, 253]
    value i= 4 and j =3 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253]]
    candidate_row : [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282]]
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 272, 619, 282], [676, 272, 708, 282]]
    len of rows top =6 ,botttom 5 , i= 4, j= 3
    top_bbox = [587, 234, 620, 282] 
    bottom_bbox = [587, 272, 619, 282]
    overlaping  
    dist :  10.0
    distance statisfy :  3
    value i= 4 and j =4 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253]]
    candidate_row : [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282]]
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 272, 619, 282], [676, 272, 708, 282]]
    len of rows top =6 ,botttom 5 , i= 4, j= 4
    top_bbox = [587, 234, 620, 282] 
    bottom_bbox = [676, 272, 708, 282]
    Not overlaping  
    (right most) adding to candidate row (top box)  [587, 234, 620, 282]
    (right most)adding to mergeList  box  [587, 234, 620, 282]
    value i= 5 and j =4 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282]]
    candidate_row : [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282]]
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 272, 619, 282], [676, 272, 708, 282]]
    len of rows top =6 ,botttom 5 , i= 5, j= 4
    top_bbox = [676, 234, 709, 282] 
    bottom_bbox = [676, 272, 708, 282]
    overlaping  
    dist :  10.0
    distance statisfy :  4
    adding *last* candidate row (top box)   [676, 234, 709, 282]
    value i= 5 and j =5 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282]]
    candidate_row : [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  5
    value i= 6 and j =5 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    candidate_row : [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282]]
    candidate_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282]]
    
    ---------------------------------------------------------------
    
    loop  6
    top_row  [[83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282]]
    bottom_row  [[83, 288, 255, 298]]
    len of rows top =5 ,botttom 1 , i= 0, j= 0
    top_bbox = [83, 272, 255, 298] 
    bottom_bbox = [83, 288, 255, 298]
    overlaping  
    dist :  10.0
    distance statisfy :  0
    adding *last* candidate row (top box)   [83, 272, 255, 298]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282]]
    candidate_row : [[83, 272, 255, 298]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =1 
    top box  is not empty :  1
    value i= 2 and j =1 
    top box  is not empty :  2
    value i= 3 and j =1 
    top box  is not empty :  3
    value i= 4 and j =1 
    top box  is not empty :  4
    value i= 5 and j =1 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282]]
    candidate_row : [[83, 272, 255, 298]]
    candidate_row  [[83, 272, 255, 298]]
    
    ---------------------------------------------------------------
    
    loop  7
    top_row  [[83, 272, 255, 298]]
    bottom_row  [[48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328]]
    len of rows top =1 ,botttom 6 , i= 0, j= 0
    top_bbox = [83, 272, 255, 298] 
    bottom_bbox = [48, 318, 55, 328]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [48, 318, 55, 328]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282]]
    candidate_row : [[48, 318, 55, 328]]
    
    --exsiting the while 1 loop--
    
    top_row  [[83, 272, 255, 298]]
    bottom_row  [[48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328]]
    len of rows top =1 ,botttom 6 , i= 0, j= 1
    top_bbox = [83, 272, 255, 298] 
    bottom_bbox = [84, 318, 268, 390]
    overlaping  
    dist :  20.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [84, 318, 268, 390]
    value i= 0 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282]]
    candidate_row : [[48, 318, 55, 328], [84, 318, 268, 390]]
    
    --exsiting the while 1 loop--
    
    top_row  [[83, 272, 255, 298]]
    bottom_row  [[48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328]]
    len of rows top =1 ,botttom 6 , i= 0, j= 2
    top_bbox = [83, 272, 255, 298] 
    bottom_bbox = [412, 318, 443, 328]
    Not overlaping  
    (right most)adding to mergeList  box  [83, 272, 255, 298]
    value i= 1 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298]]
    candidate_row : [[48, 318, 55, 328], [84, 318, 268, 390]]
    
    --exsiting the while 1 loop--
    
    breaking top row loop
    bottom bbox is not empty :  2
    value i= 1 and j =3 
    bottom bbox is not empty :  3
    value i= 1 and j =4 
    bottom bbox is not empty :  4
    value i= 1 and j =5 
    bottom bbox is not empty :  5
    value i= 1 and j =6 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298]]
    candidate_row : [[48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328]]
    candidate_row  [[48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328]]
    
    ---------------------------------------------------------------
    
    loop  8
    top_row  [[48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328]]
    bottom_row  [[84, 333, 268, 390]]
    len of rows top =6 ,botttom 1 , i= 0, j= 0
    top_bbox = [48, 318, 55, 328] 
    bottom_bbox = [84, 333, 268, 390]
    Not overlaping  
    (right most)adding to mergeList  box  [48, 318, 55, 328]
    value i= 1 and j =0 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328]]
    bottom_row  [[84, 333, 268, 390]]
    len of rows top =6 ,botttom 1 , i= 1, j= 0
    top_bbox = [84, 318, 268, 390] 
    bottom_bbox = [84, 333, 268, 390]
    overlaping  
    dist :  57.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [84, 333, 268, 390]
    value i= 1 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328]]
    candidate_row : [[84, 333, 268, 390]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =1 
    top box  is not empty :  2
    value i= 3 and j =1 
    top box  is not empty :  3
    value i= 4 and j =1 
    top box  is not empty :  4
    value i= 5 and j =1 
    top box  is not empty :  5
    value i= 6 and j =1 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328]]
    candidate_row : [[84, 333, 268, 390]]
    candidate_row  [[84, 333, 268, 390]]
    
    ---------------------------------------------------------------
    
    loop  9
    top_row  [[84, 333, 268, 390]]
    bottom_row  [[84, 349, 238, 362]]
    len of rows top =1 ,botttom 1 , i= 0, j= 0
    top_bbox = [84, 333, 268, 390] 
    bottom_bbox = [84, 349, 238, 362]
    overlaping  
    dist :  41.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [84, 349, 238, 362]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328]]
    candidate_row : [[84, 349, 238, 362]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =1 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390]]
    candidate_row : [[84, 349, 238, 362]]
    candidate_row  [[84, 349, 238, 362]]
    
    ---------------------------------------------------------------
    
    loop  10
    top_row  [[84, 349, 238, 362]]
    bottom_row  [[84, 365, 189, 378]]
    len of rows top =1 ,botttom 1 , i= 0, j= 0
    top_bbox = [84, 349, 238, 362] 
    bottom_bbox = [84, 365, 189, 378]
    overlaping  
    dist :  3.0
    distance statisfy :  0
    adding *last* candidate row (top box)   [84, 349, 238, 378]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390]]
    candidate_row : [[84, 349, 238, 378]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =1 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 238, 378]]
    candidate_row : [[84, 349, 238, 378]]
    candidate_row  [[84, 349, 238, 378]]
    
    ---------------------------------------------------------------
    
    loop  11
    top_row  [[84, 349, 238, 378]]
    bottom_row  [[83, 380, 255, 390]]
    len of rows top =1 ,botttom 1 , i= 0, j= 0
    top_bbox = [84, 349, 238, 378] 
    bottom_bbox = [83, 380, 255, 390]
    overlaping  
    dist :  2.0
    distance statisfy :  0
    adding *last* candidate row (top box)   [84, 349, 255, 390]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390]]
    candidate_row : [[84, 349, 255, 390]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =1 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390]]
    candidate_row : [[84, 349, 255, 390]]
    candidate_row  [[84, 349, 255, 390]]
    
    ---------------------------------------------------------------
    
    loop  12
    top_row  [[84, 349, 255, 390]]
    bottom_row  [[48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420]]
    len of rows top =1 ,botttom 6 , i= 0, j= 0
    top_bbox = [84, 349, 255, 390] 
    bottom_bbox = [48, 410, 55, 420]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [48, 410, 55, 420]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390]]
    candidate_row : [[48, 410, 55, 420]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 349, 255, 390]]
    bottom_row  [[48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420]]
    len of rows top =1 ,botttom 6 , i= 0, j= 1
    top_bbox = [84, 349, 255, 390] 
    bottom_bbox = [84, 410, 238, 455]
    overlaping  
    dist :  20.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [84, 410, 238, 455]
    value i= 0 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390]]
    candidate_row : [[48, 410, 55, 420], [84, 410, 238, 455]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 349, 255, 390]]
    bottom_row  [[48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420]]
    len of rows top =1 ,botttom 6 , i= 0, j= 2
    top_bbox = [84, 349, 255, 390] 
    bottom_bbox = [415, 410, 439, 420]
    Not overlaping  
    (right most)adding to mergeList  box  [84, 349, 255, 390]
    value i= 1 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390]]
    candidate_row : [[48, 410, 55, 420], [84, 410, 238, 455]]
    
    --exsiting the while 1 loop--
    
    breaking top row loop
    bottom bbox is not empty :  2
    value i= 1 and j =3 
    bottom bbox is not empty :  3
    value i= 1 and j =4 
    bottom bbox is not empty :  4
    value i= 1 and j =5 
    bottom bbox is not empty :  5
    value i= 1 and j =6 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390]]
    candidate_row : [[48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420]]
    candidate_row  [[48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420]]
    
    ---------------------------------------------------------------
    
    loop  13
    top_row  [[48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420]]
    bottom_row  [[84, 426, 238, 455]]
    len of rows top =6 ,botttom 1 , i= 0, j= 0
    top_bbox = [48, 410, 55, 420] 
    bottom_bbox = [84, 426, 238, 455]
    Not overlaping  
    (right most)adding to mergeList  box  [48, 410, 55, 420]
    value i= 1 and j =0 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420]]
    bottom_row  [[84, 426, 238, 455]]
    len of rows top =6 ,botttom 1 , i= 1, j= 0
    top_bbox = [84, 410, 238, 455] 
    bottom_bbox = [84, 426, 238, 455]
    overlaping  
    dist :  29.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [84, 426, 238, 455]
    value i= 1 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420]]
    candidate_row : [[84, 426, 238, 455]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =1 
    top box  is not empty :  2
    value i= 3 and j =1 
    top box  is not empty :  3
    value i= 4 and j =1 
    top box  is not empty :  4
    value i= 5 and j =1 
    top box  is not empty :  5
    value i= 6 and j =1 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420]]
    candidate_row : [[84, 426, 238, 455]]
    candidate_row  [[84, 426, 238, 455]]
    
    ---------------------------------------------------------------
    
    loop  14
    top_row  [[84, 426, 238, 455]]
    bottom_row  [[84, 442, 189, 455]]
    len of rows top =1 ,botttom 1 , i= 0, j= 0
    top_bbox = [84, 426, 238, 455] 
    bottom_bbox = [84, 442, 189, 455]
    overlaping  
    dist :  13.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [84, 442, 189, 455]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420]]
    candidate_row : [[84, 442, 189, 455]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =1 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455]]
    candidate_row : [[84, 442, 189, 455]]
    candidate_row  [[84, 442, 189, 455]]
    
    ---------------------------------------------------------------
    
    loop  15
    top_row  [[84, 442, 189, 455]]
    bottom_row  [[665, 488, 709, 499]]
    len of rows top =1 ,botttom 1 , i= 0, j= 0
    top_bbox = [84, 442, 189, 455] 
    bottom_bbox = [665, 488, 709, 499]
    Not overlaping  
    (right most)adding to mergeList  box  [84, 442, 189, 455]
    value i= 1 and j =0 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    breaking top row loop
    bottom bbox is not empty :  0
    value i= 1 and j =1 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455]]
    candidate_row : [[665, 488, 709, 499]]
    candidate_row  [[665, 488, 709, 499]]
    
    ---------------------------------------------------------------
    
    loop  16
    top_row  [[665, 488, 709, 499]]
    bottom_row  [[412, 513, 531, 551], [666, 513, 709, 524]]
    len of rows top =1 ,botttom 2 , i= 0, j= 0
    top_bbox = [665, 488, 709, 499] 
    bottom_bbox = [412, 513, 531, 551]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [412, 513, 531, 551]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455]]
    candidate_row : [[412, 513, 531, 551]]
    
    --exsiting the while 1 loop--
    
    top_row  [[665, 488, 709, 499]]
    bottom_row  [[412, 513, 531, 551], [666, 513, 709, 524]]
    len of rows top =1 ,botttom 2 , i= 0, j= 1
    top_bbox = [665, 488, 709, 499] 
    bottom_bbox = [666, 513, 709, 524]
    overlaping  
    dist :  14.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [666, 513, 709, 524]
    value i= 0 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455]]
    candidate_row : [[412, 513, 531, 551], [666, 513, 709, 524]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =2 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499]]
    candidate_row : [[412, 513, 531, 551], [666, 513, 709, 524]]
    candidate_row  [[412, 513, 531, 551], [666, 513, 709, 524]]
    
    ---------------------------------------------------------------
    
    loop  17
    top_row  [[412, 513, 531, 551], [666, 513, 709, 524]]
    bottom_row  [[412, 537, 531, 551], [668, 537, 709, 548]]
    len of rows top =2 ,botttom 2 , i= 0, j= 0
    top_bbox = [412, 513, 531, 551] 
    bottom_bbox = [412, 537, 531, 551]
    overlaping  
    dist :  14.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [412, 537, 531, 551]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499]]
    candidate_row : [[412, 537, 531, 551]]
    
    --exsiting the while 1 loop--
    
    top_row  [[412, 513, 531, 551], [666, 513, 709, 524]]
    bottom_row  [[412, 537, 531, 551], [668, 537, 709, 548]]
    len of rows top =2 ,botttom 2 , i= 0, j= 1
    top_bbox = [412, 513, 531, 551] 
    bottom_bbox = [668, 537, 709, 548]
    Not overlaping  
    (right most)adding to mergeList  box  [412, 513, 531, 551]
    value i= 1 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551]]
    candidate_row : [[412, 537, 531, 551]]
    
    --exsiting the while 1 loop--
    
    top_row  [[412, 513, 531, 551], [666, 513, 709, 524]]
    bottom_row  [[412, 537, 531, 551], [668, 537, 709, 548]]
    len of rows top =2 ,botttom 2 , i= 1, j= 1
    top_bbox = [666, 513, 709, 524] 
    bottom_bbox = [668, 537, 709, 548]
    overlaping  
    dist :  13.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [668, 537, 709, 548]
    value i= 1 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551]]
    candidate_row : [[412, 537, 531, 551], [668, 537, 709, 548]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =2 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524]]
    candidate_row : [[412, 537, 531, 551], [668, 537, 709, 548]]
    candidate_row  [[412, 537, 531, 551], [668, 537, 709, 548]]
    
    ---------------------------------------------------------------
    
    loop  18
    top_row  [[412, 537, 531, 551], [668, 537, 709, 548]]
    bottom_row  [[410, 562, 494, 576], [665, 562, 709, 573]]
    len of rows top =2 ,botttom 2 , i= 0, j= 0
    top_bbox = [412, 537, 531, 551] 
    bottom_bbox = [410, 562, 494, 576]
    overlaping  
    dist :  11.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [410, 562, 494, 576]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524]]
    candidate_row : [[410, 562, 494, 576]]
    
    --exsiting the while 1 loop--
    
    top_row  [[412, 537, 531, 551], [668, 537, 709, 548]]
    bottom_row  [[410, 562, 494, 576], [665, 562, 709, 573]]
    len of rows top =2 ,botttom 2 , i= 0, j= 1
    top_bbox = [412, 537, 531, 551] 
    bottom_bbox = [665, 562, 709, 573]
    Not overlaping  
    (right most)adding to mergeList  box  [412, 537, 531, 551]
    value i= 1 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524], [412, 537, 531, 551]]
    candidate_row : [[410, 562, 494, 576]]
    
    --exsiting the while 1 loop--
    
    top_row  [[412, 537, 531, 551], [668, 537, 709, 548]]
    bottom_row  [[410, 562, 494, 576], [665, 562, 709, 573]]
    len of rows top =2 ,botttom 2 , i= 1, j= 1
    top_bbox = [668, 537, 709, 548] 
    bottom_bbox = [665, 562, 709, 573]
    overlaping  
    dist :  14.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [665, 562, 709, 573]
    value i= 1 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524], [412, 537, 531, 551]]
    candidate_row : [[410, 562, 494, 576], [665, 562, 709, 573]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =2 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524], [412, 537, 531, 551], [668, 537, 709, 548]]
    candidate_row : [[410, 562, 494, 576], [665, 562, 709, 573]]
    candidate_row  [[410, 562, 494, 576], [665, 562, 709, 573]]
    
    ---------------------------------------------------------------
    
    loop  19
    top_row  [[410, 562, 494, 576], [665, 562, 709, 573]]
    bottom_row  [[42, 612, 396, 625]]
    len of rows top =2 ,botttom 1 , i= 0, j= 0
    top_bbox = [410, 562, 494, 576] 
    bottom_bbox = [42, 612, 396, 625]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [42, 612, 396, 625]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524], [412, 537, 531, 551], [668, 537, 709, 548]]
    candidate_row : [[42, 612, 396, 625]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =1 
    top box  is not empty :  1
    value i= 2 and j =1 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524], [412, 537, 531, 551], [668, 537, 709, 548], [410, 562, 494, 576], [665, 562, 709, 573]]
    candidate_row : [[42, 612, 396, 625]]
    candidate_row  [[42, 612, 396, 625]]
    
    ---------------------------------------------------------------
    
    loop  20
    top_row  [[42, 612, 396, 625]]
    bottom_row  [[42, 641, 68, 651], [110, 641, 189, 654], [72, 642, 106, 651]]
    len of rows top =1 ,botttom 3 , i= 0, j= 0
    top_bbox = [42, 612, 396, 625] 
    bottom_bbox = [42, 641, 68, 651]
    overlaping  
    dist :  16.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [42, 641, 68, 651]
    value i= 0 and j =1 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524], [412, 537, 531, 551], [668, 537, 709, 548], [410, 562, 494, 576], [665, 562, 709, 573]]
    candidate_row : [[42, 641, 68, 651]]
    
    --exsiting the while 1 loop--
    
    top_row  [[42, 612, 396, 625]]
    bottom_row  [[42, 641, 68, 651], [110, 641, 189, 654], [72, 642, 106, 651]]
    len of rows top =1 ,botttom 3 , i= 0, j= 1
    top_bbox = [42, 612, 396, 625] 
    bottom_bbox = [110, 641, 189, 654]
    overlaping  
    dist :  16.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [110, 641, 189, 654]
    value i= 0 and j =2 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524], [412, 537, 531, 551], [668, 537, 709, 548], [410, 562, 494, 576], [665, 562, 709, 573]]
    candidate_row : [[42, 641, 68, 651], [110, 641, 189, 654]]
    
    --exsiting the while 1 loop--
    
    top_row  [[42, 612, 396, 625]]
    bottom_row  [[42, 641, 68, 651], [110, 641, 189, 654], [72, 642, 106, 651]]
    len of rows top =1 ,botttom 3 , i= 0, j= 2
    top_bbox = [42, 612, 396, 625] 
    bottom_bbox = [72, 642, 106, 651]
    overlaping  
    dist :  17.0
    distance not statisfy:  2
    adding to candidate row (bottom_bbox)  [72, 642, 106, 651]
    value i= 0 and j =3 
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524], [412, 537, 531, 551], [668, 537, 709, 548], [410, 562, 494, 576], [665, 562, 709, 573]]
    candidate_row : [[42, 641, 68, 651], [110, 641, 189, 654], [72, 642, 106, 651]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =3 
    exsiting the while 2 loop
    merge_list :  [[49, 44, 368, 73], [411, 34, 483, 45], [567, 34, 679, 45], [49, 60, 368, 73], [411, 59, 498, 70], [567, 58, 635, 69], [411, 83, 476, 94], [567, 83, 638, 94], [411, 108, 520, 119], [566, 107, 687, 118], [48, 244, 56, 253], [76, 243, 147, 256], [417, 243, 438, 256], [467, 243, 498, 253], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [415, 272, 439, 282], [464, 272, 496, 282], [587, 234, 620, 282], [676, 234, 709, 282], [83, 272, 255, 298], [48, 318, 55, 328], [84, 318, 268, 390], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328], [84, 333, 268, 390], [84, 349, 255, 390], [84, 349, 255, 390], [84, 349, 255, 390], [48, 410, 55, 420], [84, 410, 238, 455], [415, 410, 439, 420], [464, 410, 496, 420], [587, 410, 619, 420], [670, 410, 708, 420], [84, 426, 238, 455], [84, 442, 189, 455], [665, 488, 709, 499], [412, 513, 531, 551], [666, 513, 709, 524], [412, 537, 531, 551], [668, 537, 709, 548], [410, 562, 494, 576], [665, 562, 709, 573], [42, 612, 396, 625]]
    candidate_row : [[42, 641, 68, 651], [110, 641, 189, 654], [72, 642, 106, 651]]
    candidate_row  [[42, 641, 68, 651], [110, 641, 189, 654], [72, 642, 106, 651]]
    
    ---------------------------------------------------------------
    
    


```python
d[-1]
```




    [72, 642, 106, 651]




```python
%debug
```

    > <ipython-input-16-b9ddf0e2337f>(33)col_clestring()
         31         while i < top_row_len:
         32             while j < bottom_row_len:
    ---> 33                 top_bbox = top_row[i]
         34                 bottom_bbox= bottom_row[j]
         35 
    
    ipdb> bottom_row_len
    6
    ipdb> top_row_len
    1
    ipdb> i,j
    (1, 2)
    ipdb> top_row
    [[83, 272, 255, 298]]
    ipdb> bottom_row
    [[48, 318, 55, 328], [76, 318, 123, 331], [412, 318, 443, 328], [464, 318, 496, 328], [587, 318, 619, 328], [669, 318, 708, 328]]
    ipdb> exit
    


```python
# visualization of image, word detected by ocr, words after removal of noise, and Graph generated.

idx=2
img=cv2.imread(file_name[idx])
# display_image(img)

word_bbox_list, bbox_width =  extract_bbox_info(img)
# display_bounding_box(img, word_bbox_list,'orginal words')

# print_ocr_string(img)

word_list = remove_noise(bbox_width,word_bbox_list)
img=cv2.imread(file_name[idx])
# display_bounding_box(img, word_list,'word after removal')

lines = Line_Formation(word_list)
word_dict , contain_dict= get_word_dict(lines)
width, hight ,_ = img.shape
lookup= get_NN_lookup_table(lines,width,hight)
adj_mat, num_feature = create_graph_numeric_feature(word_dict,lookup)

img = cv2.imread(file_name[idx])
# display_connected_graph(img, lookup, 'merge',True)

cv2.waitKey(0)
# feature_vector = create_feature_vector(word_dict,num_feature)
```




    -1




```python
# a,b = row_merge(lines,30)
a,b,c = row_clustring(lines,10)
# idx=1
img=cv2.imread(file_name[idx])
display_bounding_box(img, a, 'after row clusting' )
d= col_clestring(c,10)
img=cv2.imread(file_name[idx])
display_bounding_box(img, d,'after col clusting' )
cv2.waitKey(0)
```

    [['word_1_1', 'Acme', [927, 88, 985, 104]], ['word_1_3', 'Queen', [1057, 88, 1123, 108]], ['word_1_4', 'Street', [1132, 88, 1192, 104]]]
    total word =  3
    curr_bbox :  [927, 88, 985, 104]
    next_bbox :  ['word_1_3', 'Queen', [1057, 88, 1123, 108]]
    dist  72.0
    row_bbox_list (inserting) [927, 88, 985, 104]
    curr_bbox (adding) [1057, 88, 1123, 108]
    next_bbox_id  2
    next_bbox :  ['word_1_4', 'Street', [1132, 88, 1192, 104]]
    dist  9.0
    curr_bbox (merging) [1057, 88, 1192, 108]
    next_bbox_id  3
    row_bbox_list (inserting) after merge [1057, 88, 1192, 108]
    ---------------------------line 1-----------------------------
    [['word_1_9', '§', [100, 116, 176, 268]], ['word_1_5', '4161234567', [858, 117, 985, 133]], ['word_1_6', 'Hamlet,', [1034, 116, 1109, 136]], ['word_1_7', 'Ontario', [1118, 116, 1192, 133]]]
    total word =  4
    curr_bbox :  [100, 116, 176, 268]
    next_bbox :  ['word_1_5', '4161234567', [858, 117, 985, 133]]
    dist  682.0
    row_bbox_list (inserting) [100, 116, 176, 268]
    curr_bbox (adding) [858, 117, 985, 133]
    next_bbox_id  2
    next_bbox :  ['word_1_6', 'Hamlet,', [1034, 116, 1109, 136]]
    dist  49.0
    row_bbox_list (inserting) [858, 117, 985, 133]
    curr_bbox (adding) [1034, 116, 1109, 136]
    next_bbox_id  3
    next_bbox :  ['word_1_7', 'Ontario', [1118, 116, 1192, 133]]
    dist  9.0
    curr_bbox (merging) [1034, 116, 1192, 136]
    next_bbox_id  4
    row_bbox_list (inserting) after merge [1034, 116, 1192, 136]
    ---------------------------line 2-----------------------------
    [['word_1_8', 'N2R2R2', [1107, 147, 1192, 163]]]
    total word =  1
    curr_bbox :  [1107, 147, 1192, 163]
    row_bbox_list (inserting) after merge [1107, 147, 1192, 163]
    ---------------------------line 3-----------------------------
    [['word_1_11', '“sn', [1113, 175, 1193, 192]]]
    total word =  1
    curr_bbox :  [1113, 175, 1193, 192]
    row_bbox_list (inserting) after merge [1113, 175, 1193, 192]
    ---------------------------line 4-----------------------------
    [['word_1_12', 'Billed', [84, 422, 136, 438]], ['word_1_14', 'Date', [405, 422, 450, 438]], ['word_1_16', 'Issue', [485, 422, 536, 438]], ['word_1_17', 'Invoice', [649, 422, 719, 438]], ['word_1_18', 'Number', [728, 422, 807, 438]], ['word_1_19', 'Amount', [995, 422, 1074, 438]], ['word_1_20', 'Due', [1083, 422, 1122, 438]], ['word_1_21', '(CAD)', [1130, 422, 1191, 443]]]
    total word =  8
    curr_bbox :  [84, 422, 136, 438]
    next_bbox :  ['word_1_14', 'Date', [405, 422, 450, 438]]
    dist  269.0
    row_bbox_list (inserting) [84, 422, 136, 438]
    curr_bbox (adding) [405, 422, 450, 438]
    next_bbox_id  2
    next_bbox :  ['word_1_16', 'Issue', [485, 422, 536, 438]]
    dist  35.0
    row_bbox_list (inserting) [405, 422, 450, 438]
    curr_bbox (adding) [485, 422, 536, 438]
    next_bbox_id  3
    next_bbox :  ['word_1_17', 'Invoice', [649, 422, 719, 438]]
    dist  113.0
    row_bbox_list (inserting) [485, 422, 536, 438]
    curr_bbox (adding) [649, 422, 719, 438]
    next_bbox_id  4
    next_bbox :  ['word_1_18', 'Number', [728, 422, 807, 438]]
    dist  9.0
    curr_bbox (merging) [649, 422, 807, 438]
    next_bbox_id  5
    next_bbox :  ['word_1_19', 'Amount', [995, 422, 1074, 438]]
    dist  188.0
    row_bbox_list (inserting) [649, 422, 807, 438]
    curr_bbox (adding) [995, 422, 1074, 438]
    next_bbox_id  6
    next_bbox :  ['word_1_20', 'Due', [1083, 422, 1122, 438]]
    dist  9.0
    curr_bbox (merging) [995, 422, 1122, 438]
    next_bbox_id  7
    next_bbox :  ['word_1_21', '(CAD)', [1130, 422, 1191, 443]]
    dist  8.0
    curr_bbox (merging) [995, 422, 1191, 443]
    next_bbox_id  8
    row_bbox_list (inserting) after merge [995, 422, 1191, 443]
    ---------------------------line 5-----------------------------
    [['word_1_22', 'vanapeop', [82, 450, 245, 501]], ['word_1_23', '06/03/2019', [404, 450, 516, 467]], ['word_1_24', '0000005', [648, 451, 736, 467]], ['word_1_25', '$5,', [932, 446, 1007, 506]], ['word_1_28', 'O.', [1080, 453, 1122, 495]]]
    total word =  5
    curr_bbox :  [82, 450, 245, 501]
    next_bbox :  ['word_1_23', '06/03/2019', [404, 450, 516, 467]]
    dist  159.0
    row_bbox_list (inserting) [82, 450, 245, 501]
    curr_bbox (adding) [404, 450, 516, 467]
    next_bbox_id  2
    next_bbox :  ['word_1_24', '0000005', [648, 451, 736, 467]]
    dist  132.0
    row_bbox_list (inserting) [404, 450, 516, 467]
    curr_bbox (adding) [648, 451, 736, 467]
    next_bbox_id  3
    next_bbox :  ['word_1_25', '$5,', [932, 446, 1007, 506]]
    dist  196.0
    row_bbox_list (inserting) [648, 451, 736, 467]
    curr_bbox (adding) [932, 446, 1007, 506]
    next_bbox_id  4
    next_bbox :  ['word_1_28', 'O.', [1080, 453, 1122, 495]]
    dist  73.0
    row_bbox_list (inserting) [932, 446, 1007, 506]
    curr_bbox (adding) [1080, 453, 1122, 495]
    next_bbox_id  5
    row_bbox_list (inserting) after last word ['word_1_28', 'O.', [1080, 453, 1122, 495]]
    ---------------------------line 6-----------------------------
    [['word_1_31', '123', [84, 509, 119, 525]], ['word_1_32', 'Main', [128, 508, 174, 525]], ['word_1_33', 'Street', [183, 509, 243, 525]], ['word_1_37', 'Due', [405, 521, 444, 537]], ['word_1_38', 'Date', [453, 521, 498, 537]]]
    total word =  5
    curr_bbox :  [84, 509, 119, 525]
    next_bbox :  ['word_1_32', 'Main', [128, 508, 174, 525]]
    dist  9.0
    curr_bbox (merging) [84, 509, 174, 525]
    next_bbox_id  2
    next_bbox :  ['word_1_33', 'Street', [183, 509, 243, 525]]
    dist  9.0
    curr_bbox (merging) [84, 509, 243, 525]
    next_bbox_id  3
    next_bbox :  ['word_1_37', 'Due', [405, 521, 444, 537]]
    dist  162.0
    row_bbox_list (inserting) [84, 509, 243, 525]
    curr_bbox (adding) [405, 521, 444, 537]
    next_bbox_id  4
    next_bbox :  ['word_1_38', 'Date', [453, 521, 498, 537]]
    dist  9.0
    curr_bbox (merging) [405, 521, 498, 537]
    next_bbox_id  5
    row_bbox_list (inserting) after merge [405, 521, 498, 537]
    ---------------------------line 7-----------------------------
    [['word_1_39', 'Townsville,', [82, 538, 190, 558]], ['word_1_40', 'Ontario', [200, 539, 273, 555]], ['word_1_41', '07/03/2019', [404, 549, 516, 566]]]
    total word =  3
    curr_bbox :  [82, 538, 190, 558]
    next_bbox :  ['word_1_40', 'Ontario', [200, 539, 273, 555]]
    dist  10.0
    curr_bbox (merging) [82, 538, 273, 558]
    next_bbox_id  2
    next_bbox :  ['word_1_41', '07/03/2019', [404, 549, 516, 566]]
    dist  131.0
    row_bbox_list (inserting) [82, 538, 273, 558]
    curr_bbox (adding) [404, 549, 516, 566]
    next_bbox_id  3
    row_bbox_list (inserting) after last word ['word_1_41', '07/03/2019', [404, 549, 516, 566]]
    ---------------------------line 8-----------------------------
    [['word_1_42', 'M4L2DY', [84, 568, 169, 584]]]
    total word =  1
    curr_bbox :  [84, 568, 169, 584]
    row_bbox_list (inserting) after last word ['word_1_42', 'M4L2DY', [84, 568, 169, 584]]
    ---------------------------line 9-----------------------------
    [['word_1_43', 'Description', [84, 730, 194, 751]], ['word_1_44', 'Rate', [819, 726, 865, 755]], ['word_1_45', 'Qty', [994, 730, 1029, 751]], ['word_1_46', 'Line', [1096, 730, 1137, 746]], ['word_1_47', 'Total', [1145, 730, 1191, 746]]]
    total word =  5
    curr_bbox :  [84, 730, 194, 751]
    next_bbox :  ['word_1_44', 'Rate', [819, 726, 865, 755]]
    dist  625.0
    row_bbox_list (inserting) [84, 730, 194, 751]
    curr_bbox (adding) [819, 726, 865, 755]
    next_bbox_id  2
    next_bbox :  ['word_1_45', 'Qty', [994, 730, 1029, 751]]
    dist  129.0
    row_bbox_list (inserting) [819, 726, 865, 755]
    curr_bbox (adding) [994, 730, 1029, 751]
    next_bbox_id  3
    next_bbox :  ['word_1_46', 'Line', [1096, 730, 1137, 746]]
    dist  67.0
    row_bbox_list (inserting) [994, 730, 1029, 751]
    curr_bbox (adding) [1096, 730, 1137, 746]
    next_bbox_id  4
    next_bbox :  ['word_1_47', 'Total', [1145, 730, 1191, 746]]
    dist  8.0
    curr_bbox (merging) [1096, 730, 1191, 746]
    next_bbox_id  5
    row_bbox_list (inserting) after merge [1096, 730, 1191, 746]
    ---------------------------line 10-----------------------------
    [['word_1_48', 'Project', [84, 791, 153, 813]], ['word_1_49', '$5,000.00', [764, 791, 865, 811]], ['word_1_51', '$5,000.00', [1091, 791, 1192, 811]]]
    total word =  3
    curr_bbox :  [84, 791, 153, 813]
    next_bbox :  ['word_1_49', '$5,000.00', [764, 791, 865, 811]]
    dist  611.0
    row_bbox_list (inserting) [84, 791, 153, 813]
    curr_bbox (adding) [764, 791, 865, 811]
    next_bbox_id  2
    next_bbox :  ['word_1_51', '$5,000.00', [1091, 791, 1192, 811]]
    dist  226.0
    row_bbox_list (inserting) [764, 791, 865, 811]
    curr_bbox (adding) [1091, 791, 1192, 811]
    next_bbox_id  3
    row_bbox_list (inserting) after last word ['word_1_51', '$5,000.00', [1091, 791, 1192, 811]]
    ---------------------------line 11-----------------------------
    [['word_1_52', 'Expenses', [84, 855, 182, 876]], ['word_1_53', '$500.00', [783, 854, 865, 873]], ['word_1_55', '$500.00', [1110, 854, 1192, 873]]]
    total word =  3
    curr_bbox :  [84, 855, 182, 876]
    next_bbox :  ['word_1_53', '$500.00', [783, 854, 865, 873]]
    dist  601.0
    row_bbox_list (inserting) [84, 855, 182, 876]
    curr_bbox (adding) [783, 854, 865, 873]
    next_bbox_id  2
    next_bbox :  ['word_1_55', '$500.00', [1110, 854, 1192, 873]]
    dist  245.0
    row_bbox_list (inserting) [783, 854, 865, 873]
    curr_bbox (adding) [1110, 854, 1192, 873]
    next_bbox_id  3
    row_bbox_list (inserting) after last word ['word_1_55', '$500.00', [1110, 854, 1192, 873]]
    ---------------------------line 12-----------------------------
    [['word_1_56', 'Subtotal', [878, 954, 960, 984]], ['word_1_57', '5,500.00', [1104, 959, 1192, 978]]]
    total word =  2
    curr_bbox :  [878, 954, 960, 984]
    next_bbox :  ['word_1_57', '5,500.00', [1104, 959, 1192, 978]]
    dist  144.0
    row_bbox_list (inserting) [878, 954, 960, 984]
    curr_bbox (adding) [1104, 959, 1192, 978]
    next_bbox_id  2
    row_bbox_list (inserting) after last word ['word_1_57', '5,500.00', [1104, 959, 1192, 978]]
    ---------------------------line 13-----------------------------
    [['word_1_58', 'Tax', [926, 993, 961, 1022]], ['word_1_59', '0.00', [1149, 997, 1192, 1013]]]
    total word =  2
    curr_bbox :  [926, 993, 961, 1022]
    next_bbox :  ['word_1_59', '0.00', [1149, 997, 1192, 1013]]
    dist  188.0
    row_bbox_list (inserting) [926, 993, 961, 1022]
    curr_bbox (adding) [1149, 997, 1192, 1013]
    next_bbox_id  2
    row_bbox_list (inserting) after last word ['word_1_59', '0.00', [1149, 997, 1192, 1013]]
    ---------------------------line 14-----------------------------
    [['word_1_60', 'Total', [913, 1055, 960, 1085]], ['word_1_61', '5,500.00', [1104, 1060, 1192, 1079]]]
    total word =  2
    curr_bbox :  [913, 1055, 960, 1085]
    next_bbox :  ['word_1_61', '5,500.00', [1104, 1060, 1192, 1079]]
    dist  144.0
    row_bbox_list (inserting) [913, 1055, 960, 1085]
    curr_bbox (adding) [1104, 1060, 1192, 1079]
    next_bbox_id  2
    row_bbox_list (inserting) after last word ['word_1_61', '5,500.00', [1104, 1060, 1192, 1079]]
    ---------------------------line 15-----------------------------
    [['word_1_62', 'Amount', [830, 1089, 909, 1105]], ['word_1_63', 'Paid', [918, 1088, 960, 1105]], ['word_1_64', '0.00', [1149, 1089, 1192, 1105]]]
    total word =  3
    curr_bbox :  [830, 1089, 909, 1105]
    next_bbox :  ['word_1_63', 'Paid', [918, 1088, 960, 1105]]
    dist  9.0
    curr_bbox (merging) [830, 1089, 960, 1105]
    next_bbox_id  2
    next_bbox :  ['word_1_64', '0.00', [1149, 1089, 1192, 1105]]
    dist  189.0
    row_bbox_list (inserting) [830, 1089, 960, 1105]
    curr_bbox (adding) [1149, 1089, 1192, 1105]
    next_bbox_id  3
    row_bbox_list (inserting) after last word ['word_1_64', '0.00', [1149, 1089, 1192, 1105]]
    ---------------------------line 16-----------------------------
    [['word_1_68', '$5,500.00', [1091, 1151, 1192, 1171]], ['word_1_65', 'Amount', [764, 1152, 843, 1168]], ['word_1_66', 'Due', [852, 1152, 891, 1168]], ['word_1_67', '(CAD)', [899, 1152, 960, 1173]]]
    total word =  4
    curr_bbox :  [1091, 1151, 1192, 1171]
    next_bbox :  ['word_1_65', 'Amount', [764, 1152, 843, 1168]]
    dist  428.0
    row_bbox_list (inserting) [1091, 1151, 1192, 1171]
    curr_bbox (adding) [764, 1152, 843, 1168]
    next_bbox_id  2
    next_bbox :  ['word_1_66', 'Due', [852, 1152, 891, 1168]]
    dist  9.0
    curr_bbox (merging) [764, 1152, 891, 1168]
    next_bbox_id  3
    next_bbox :  ['word_1_67', '(CAD)', [899, 1152, 960, 1173]]
    dist  8.0
    curr_bbox (merging) [764, 1152, 960, 1173]
    next_bbox_id  4
    row_bbox_list (inserting) after merge [764, 1152, 960, 1173]
    ---------------------------line 17-----------------------------
    enter for calculaton 
    loop  1
    top_row  [[927, 88, 985, 104], [1057, 88, 1192, 108]]
    bottom_row  [[100, 116, 176, 268], [858, 117, 985, 133], [1034, 116, 1192, 136]]
    len of rows top =2 ,botttom 3 , i= 0, j= 0
    top_bbox = [927, 88, 985, 104] 
    bottom_bbox = [100, 116, 176, 268]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [100, 116, 176, 268]
    value i= 0 and j =1 
    merge_list :  []
    candidate_row : [[100, 116, 176, 268]]
    
    --exsiting the while 1 loop--
    
    top_row  [[927, 88, 985, 104], [1057, 88, 1192, 108]]
    bottom_row  [[100, 116, 176, 268], [858, 117, 985, 133], [1034, 116, 1192, 136]]
    len of rows top =2 ,botttom 3 , i= 0, j= 1
    top_bbox = [927, 88, 985, 104] 
    bottom_bbox = [858, 117, 985, 133]
    overlaping  
    dist :  13.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [858, 117, 985, 133]
    value i= 0 and j =2 
    merge_list :  []
    candidate_row : [[100, 116, 176, 268], [858, 117, 985, 133]]
    
    --exsiting the while 1 loop--
    
    top_row  [[927, 88, 985, 104], [1057, 88, 1192, 108]]
    bottom_row  [[100, 116, 176, 268], [858, 117, 985, 133], [1034, 116, 1192, 136]]
    len of rows top =2 ,botttom 3 , i= 0, j= 2
    top_bbox = [927, 88, 985, 104] 
    bottom_bbox = [1034, 116, 1192, 136]
    Not overlaping  
    (right most)adding to mergeList  box  [927, 88, 985, 104]
    value i= 1 and j =2 
    merge_list :  [[927, 88, 985, 104]]
    candidate_row : [[100, 116, 176, 268], [858, 117, 985, 133]]
    
    --exsiting the while 1 loop--
    
    top_row  [[927, 88, 985, 104], [1057, 88, 1192, 108]]
    bottom_row  [[100, 116, 176, 268], [858, 117, 985, 133], [1034, 116, 1192, 136]]
    len of rows top =2 ,botttom 3 , i= 1, j= 2
    top_bbox = [1057, 88, 1192, 108] 
    bottom_bbox = [1034, 116, 1192, 136]
    overlaping  
    dist :  8.0
    distance statisfy :  2
    adding *last* candidate row (top box)   [1034, 88, 1192, 136]
    value i= 1 and j =3 
    merge_list :  [[927, 88, 985, 104]]
    candidate_row : [[100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =3 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136]]
    candidate_row : [[100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136]]
    candidate_row  [[100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136]]
    
    ---------------------------------------------------------------
    
    loop  2
    top_row  [[100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136]]
    bottom_row  [[1107, 147, 1192, 163]]
    len of rows top =3 ,botttom 1 , i= 0, j= 0
    top_bbox = [100, 116, 176, 268] 
    bottom_bbox = [1107, 147, 1192, 163]
    Not overlaping  
    (right most)adding to mergeList  box  [100, 116, 176, 268]
    value i= 1 and j =0 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136]]
    bottom_row  [[1107, 147, 1192, 163]]
    len of rows top =3 ,botttom 1 , i= 1, j= 0
    top_bbox = [858, 117, 985, 133] 
    bottom_bbox = [1107, 147, 1192, 163]
    Not overlaping  
    (right most)adding to mergeList  box  [858, 117, 985, 133]
    value i= 2 and j =0 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136]]
    bottom_row  [[1107, 147, 1192, 163]]
    len of rows top =3 ,botttom 1 , i= 2, j= 0
    top_bbox = [1034, 88, 1192, 136] 
    bottom_bbox = [1107, 147, 1192, 163]
    overlaping  
    dist :  11.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [1107, 147, 1192, 163]
    value i= 2 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133]]
    candidate_row : [[1107, 147, 1192, 163]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  2
    value i= 3 and j =1 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136]]
    candidate_row : [[1107, 147, 1192, 163]]
    candidate_row  [[1107, 147, 1192, 163]]
    
    ---------------------------------------------------------------
    
    loop  3
    top_row  [[1107, 147, 1192, 163]]
    bottom_row  [[1113, 175, 1193, 192]]
    len of rows top =1 ,botttom 1 , i= 0, j= 0
    top_bbox = [1107, 147, 1192, 163] 
    bottom_bbox = [1113, 175, 1193, 192]
    overlaping  
    dist :  12.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [1113, 175, 1193, 192]
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136]]
    candidate_row : [[1113, 175, 1193, 192]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =1 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163]]
    candidate_row : [[1113, 175, 1193, 192]]
    candidate_row  [[1113, 175, 1193, 192]]
    
    ---------------------------------------------------------------
    
    loop  4
    top_row  [[1113, 175, 1193, 192]]
    bottom_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    len of rows top =1 ,botttom 5 , i= 0, j= 0
    top_bbox = [1113, 175, 1193, 192] 
    bottom_bbox = [84, 422, 136, 438]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [84, 422, 136, 438]
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163]]
    candidate_row : [[84, 422, 136, 438]]
    
    --exsiting the while 1 loop--
    
    top_row  [[1113, 175, 1193, 192]]
    bottom_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    len of rows top =1 ,botttom 5 , i= 0, j= 1
    top_bbox = [1113, 175, 1193, 192] 
    bottom_bbox = [405, 422, 450, 438]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [405, 422, 450, 438]
    value i= 0 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163]]
    candidate_row : [[84, 422, 136, 438], [405, 422, 450, 438]]
    
    --exsiting the while 1 loop--
    
    top_row  [[1113, 175, 1193, 192]]
    bottom_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    len of rows top =1 ,botttom 5 , i= 0, j= 2
    top_bbox = [1113, 175, 1193, 192] 
    bottom_bbox = [485, 422, 536, 438]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [485, 422, 536, 438]
    value i= 0 and j =3 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163]]
    candidate_row : [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438]]
    
    --exsiting the while 1 loop--
    
    top_row  [[1113, 175, 1193, 192]]
    bottom_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    len of rows top =1 ,botttom 5 , i= 0, j= 3
    top_bbox = [1113, 175, 1193, 192] 
    bottom_bbox = [649, 422, 807, 438]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [649, 422, 807, 438]
    value i= 0 and j =4 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163]]
    candidate_row : [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438]]
    
    --exsiting the while 1 loop--
    
    top_row  [[1113, 175, 1193, 192]]
    bottom_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    len of rows top =1 ,botttom 5 , i= 0, j= 4
    top_bbox = [1113, 175, 1193, 192] 
    bottom_bbox = [995, 422, 1191, 443]
    overlaping  
    dist :  230.0
    distance not statisfy:  4
    adding to candidate row (bottom_bbox)  [995, 422, 1191, 443]
    value i= 0 and j =5 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163]]
    candidate_row : [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =5 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192]]
    candidate_row : [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    candidate_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    
    ---------------------------------------------------------------
    
    loop  5
    top_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    bottom_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [932, 446, 1007, 506], [1080, 453, 1122, 495]]
    len of rows top =5 ,botttom 5 , i= 0, j= 0
    top_bbox = [84, 422, 136, 438] 
    bottom_bbox = [82, 450, 245, 501]
    overlaping  
    dist :  12.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [82, 450, 245, 501]
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192]]
    candidate_row : [[82, 450, 245, 501]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    bottom_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [932, 446, 1007, 506], [1080, 453, 1122, 495]]
    len of rows top =5 ,botttom 5 , i= 0, j= 1
    top_bbox = [84, 422, 136, 438] 
    bottom_bbox = [404, 450, 516, 467]
    Not overlaping  
    (right most)adding to mergeList  box  [84, 422, 136, 438]
    value i= 1 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438]]
    candidate_row : [[82, 450, 245, 501]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    bottom_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [932, 446, 1007, 506], [1080, 453, 1122, 495]]
    len of rows top =5 ,botttom 5 , i= 1, j= 1
    top_bbox = [405, 422, 450, 438] 
    bottom_bbox = [404, 450, 516, 467]
    overlaping  
    dist :  12.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [404, 450, 516, 467]
    value i= 1 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438]]
    candidate_row : [[82, 450, 245, 501], [404, 450, 516, 467]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    bottom_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [932, 446, 1007, 506], [1080, 453, 1122, 495]]
    len of rows top =5 ,botttom 5 , i= 1, j= 2
    top_bbox = [405, 422, 450, 438] 
    bottom_bbox = [648, 451, 736, 467]
    Not overlaping  
    (right most)adding to mergeList  box  [405, 422, 450, 438]
    value i= 2 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438]]
    candidate_row : [[82, 450, 245, 501], [404, 450, 516, 467]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    bottom_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [932, 446, 1007, 506], [1080, 453, 1122, 495]]
    len of rows top =5 ,botttom 5 , i= 2, j= 2
    top_bbox = [485, 422, 536, 438] 
    bottom_bbox = [648, 451, 736, 467]
    Not overlaping  
    (right most)adding to mergeList  box  [485, 422, 536, 438]
    value i= 3 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438]]
    candidate_row : [[82, 450, 245, 501], [404, 450, 516, 467]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    bottom_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [932, 446, 1007, 506], [1080, 453, 1122, 495]]
    len of rows top =5 ,botttom 5 , i= 3, j= 2
    top_bbox = [649, 422, 807, 438] 
    bottom_bbox = [648, 451, 736, 467]
    overlaping  
    dist :  13.0
    distance not statisfy:  2
    adding to candidate row (bottom_bbox)  [648, 451, 736, 467]
    value i= 3 and j =3 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438]]
    candidate_row : [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    bottom_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [932, 446, 1007, 506], [1080, 453, 1122, 495]]
    len of rows top =5 ,botttom 5 , i= 3, j= 3
    top_bbox = [649, 422, 807, 438] 
    bottom_bbox = [932, 446, 1007, 506]
    Not overlaping  
    (right most)adding to mergeList  box  [649, 422, 807, 438]
    value i= 4 and j =3 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438]]
    candidate_row : [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [995, 422, 1191, 443]]
    bottom_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [932, 446, 1007, 506], [1080, 453, 1122, 495]]
    len of rows top =5 ,botttom 5 , i= 4, j= 3
    top_bbox = [995, 422, 1191, 443] 
    bottom_bbox = [932, 446, 1007, 506]
    overlaping  
    dist :  3.0
    distance statisfy :  3
    value i= 4 and j =4 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438]]
    candidate_row : [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506]]
    bottom_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [932, 446, 1007, 506], [1080, 453, 1122, 495]]
    len of rows top =5 ,botttom 5 , i= 4, j= 4
    top_bbox = [932, 422, 1191, 506] 
    bottom_bbox = [1080, 453, 1122, 495]
    overlaping  
    dist :  53.0
    distance not statisfy:  4
    adding to candidate row (bottom_bbox)  [1080, 453, 1122, 495]
    value i= 4 and j =5 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438]]
    candidate_row : [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  4
    value i= 5 and j =5 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506]]
    candidate_row : [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495]]
    candidate_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495]]
    
    ---------------------------------------------------------------
    
    loop  6
    top_row  [[82, 450, 245, 501], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495]]
    bottom_row  [[84, 509, 243, 525], [405, 521, 498, 537]]
    len of rows top =4 ,botttom 2 , i= 0, j= 0
    top_bbox = [82, 450, 245, 501] 
    bottom_bbox = [84, 509, 243, 525]
    overlaping  
    dist :  8.0
    distance statisfy :  0
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495]]
    bottom_row  [[84, 509, 243, 525], [405, 521, 498, 537]]
    len of rows top =4 ,botttom 2 , i= 0, j= 1
    top_bbox = [82, 450, 245, 525] 
    bottom_bbox = [405, 521, 498, 537]
    Not overlaping  
    (right most) adding to candidate row (top box)  [82, 450, 245, 525]
    (right most)adding to mergeList  box  [82, 450, 245, 525]
    value i= 1 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525]]
    candidate_row : [[82, 450, 245, 525]]
    
    --exsiting the while 1 loop--
    
    top_row  [[82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495]]
    bottom_row  [[84, 509, 243, 525], [405, 521, 498, 537]]
    len of rows top =4 ,botttom 2 , i= 1, j= 1
    top_bbox = [404, 450, 516, 467] 
    bottom_bbox = [405, 521, 498, 537]
    overlaping  
    dist :  54.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [405, 521, 498, 537]
    value i= 1 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525]]
    candidate_row : [[82, 450, 245, 525], [405, 521, 498, 537]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =2 
    top box  is not empty :  2
    value i= 3 and j =2 
    top box  is not empty :  3
    value i= 4 and j =2 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495]]
    candidate_row : [[82, 450, 245, 525], [405, 521, 498, 537]]
    candidate_row  [[82, 450, 245, 525], [405, 521, 498, 537]]
    
    ---------------------------------------------------------------
    
    loop  7
    top_row  [[82, 450, 245, 525], [405, 521, 498, 537]]
    bottom_row  [[82, 538, 273, 558], [404, 549, 516, 566]]
    len of rows top =2 ,botttom 2 , i= 0, j= 0
    top_bbox = [82, 450, 245, 525] 
    bottom_bbox = [82, 538, 273, 558]
    overlaping  
    dist :  13.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [82, 538, 273, 558]
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495]]
    candidate_row : [[82, 538, 273, 558]]
    
    --exsiting the while 1 loop--
    
    top_row  [[82, 450, 245, 525], [405, 521, 498, 537]]
    bottom_row  [[82, 538, 273, 558], [404, 549, 516, 566]]
    len of rows top =2 ,botttom 2 , i= 0, j= 1
    top_bbox = [82, 450, 245, 525] 
    bottom_bbox = [404, 549, 516, 566]
    Not overlaping  
    (right most)adding to mergeList  box  [82, 450, 245, 525]
    value i= 1 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525]]
    candidate_row : [[82, 538, 273, 558]]
    
    --exsiting the while 1 loop--
    
    top_row  [[82, 450, 245, 525], [405, 521, 498, 537]]
    bottom_row  [[82, 538, 273, 558], [404, 549, 516, 566]]
    len of rows top =2 ,botttom 2 , i= 1, j= 1
    top_bbox = [405, 521, 498, 537] 
    bottom_bbox = [404, 549, 516, 566]
    overlaping  
    dist :  12.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [404, 549, 516, 566]
    value i= 1 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525]]
    candidate_row : [[82, 538, 273, 558], [404, 549, 516, 566]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =2 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537]]
    candidate_row : [[82, 538, 273, 558], [404, 549, 516, 566]]
    candidate_row  [[82, 538, 273, 558], [404, 549, 516, 566]]
    
    ---------------------------------------------------------------
    
    loop  8
    top_row  [[82, 538, 273, 558], [404, 549, 516, 566]]
    bottom_row  [[84, 568, 169, 584]]
    len of rows top =2 ,botttom 1 , i= 0, j= 0
    top_bbox = [82, 538, 273, 558] 
    bottom_bbox = [84, 568, 169, 584]
    overlaping  
    dist :  10.0
    distance statisfy :  0
    adding *last* candidate row (top box)   [82, 538, 273, 584]
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537]]
    candidate_row : [[82, 538, 273, 584]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  0
    value i= 1 and j =1 
    top box  is not empty :  1
    value i= 2 and j =1 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566]]
    candidate_row : [[82, 538, 273, 584]]
    candidate_row  [[82, 538, 273, 584]]
    
    ---------------------------------------------------------------
    
    loop  9
    top_row  [[82, 538, 273, 584]]
    bottom_row  [[84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    len of rows top =1 ,botttom 4 , i= 0, j= 0
    top_bbox = [82, 538, 273, 584] 
    bottom_bbox = [84, 730, 194, 751]
    overlaping  
    dist :  146.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [84, 730, 194, 751]
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566]]
    candidate_row : [[84, 730, 194, 751]]
    
    --exsiting the while 1 loop--
    
    top_row  [[82, 538, 273, 584]]
    bottom_row  [[84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    len of rows top =1 ,botttom 4 , i= 0, j= 1
    top_bbox = [82, 538, 273, 584] 
    bottom_bbox = [819, 726, 865, 755]
    Not overlaping  
    (right most)adding to mergeList  box  [82, 538, 273, 584]
    value i= 1 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584]]
    candidate_row : [[84, 730, 194, 751]]
    
    --exsiting the while 1 loop--
    
    breaking top row loop
    bottom bbox is not empty :  1
    value i= 1 and j =2 
    bottom bbox is not empty :  2
    value i= 1 and j =3 
    bottom bbox is not empty :  3
    value i= 1 and j =4 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584]]
    candidate_row : [[84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    candidate_row  [[84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    
    ---------------------------------------------------------------
    
    loop  10
    top_row  [[84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    bottom_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    len of rows top =4 ,botttom 3 , i= 0, j= 0
    top_bbox = [84, 730, 194, 751] 
    bottom_bbox = [84, 791, 153, 813]
    overlaping  
    dist :  40.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [84, 791, 153, 813]
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584]]
    candidate_row : [[84, 791, 153, 813]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    bottom_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    len of rows top =4 ,botttom 3 , i= 0, j= 1
    top_bbox = [84, 730, 194, 751] 
    bottom_bbox = [764, 791, 865, 811]
    Not overlaping  
    (right most)adding to mergeList  box  [84, 730, 194, 751]
    value i= 1 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751]]
    candidate_row : [[84, 791, 153, 813]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    bottom_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    len of rows top =4 ,botttom 3 , i= 1, j= 1
    top_bbox = [819, 726, 865, 755] 
    bottom_bbox = [764, 791, 865, 811]
    overlaping  
    dist :  36.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [764, 791, 865, 811]
    value i= 1 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751]]
    candidate_row : [[84, 791, 153, 813], [764, 791, 865, 811]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    bottom_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    len of rows top =4 ,botttom 3 , i= 1, j= 2
    top_bbox = [819, 726, 865, 755] 
    bottom_bbox = [1091, 791, 1192, 811]
    Not overlaping  
    (right most)adding to mergeList  box  [819, 726, 865, 755]
    value i= 2 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755]]
    candidate_row : [[84, 791, 153, 813], [764, 791, 865, 811]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    bottom_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    len of rows top =4 ,botttom 3 , i= 2, j= 2
    top_bbox = [994, 730, 1029, 751] 
    bottom_bbox = [1091, 791, 1192, 811]
    Not overlaping  
    (right most)adding to mergeList  box  [994, 730, 1029, 751]
    value i= 3 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751]]
    candidate_row : [[84, 791, 153, 813], [764, 791, 865, 811]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    bottom_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    len of rows top =4 ,botttom 3 , i= 3, j= 2
    top_bbox = [1096, 730, 1191, 746] 
    bottom_bbox = [1091, 791, 1192, 811]
    overlaping  
    dist :  45.0
    distance not statisfy:  2
    adding to candidate row (bottom_bbox)  [1091, 791, 1192, 811]
    value i= 3 and j =3 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751]]
    candidate_row : [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  3
    value i= 4 and j =3 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    candidate_row : [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    candidate_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    
    ---------------------------------------------------------------
    
    loop  11
    top_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    bottom_row  [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    len of rows top =3 ,botttom 3 , i= 0, j= 0
    top_bbox = [84, 791, 153, 813] 
    bottom_bbox = [84, 855, 182, 876]
    overlaping  
    dist :  42.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [84, 855, 182, 876]
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746]]
    candidate_row : [[84, 855, 182, 876]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    bottom_row  [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    len of rows top =3 ,botttom 3 , i= 0, j= 1
    top_bbox = [84, 791, 153, 813] 
    bottom_bbox = [783, 854, 865, 873]
    Not overlaping  
    (right most)adding to mergeList  box  [84, 791, 153, 813]
    value i= 1 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813]]
    candidate_row : [[84, 855, 182, 876]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    bottom_row  [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    len of rows top =3 ,botttom 3 , i= 1, j= 1
    top_bbox = [764, 791, 865, 811] 
    bottom_bbox = [783, 854, 865, 873]
    overlaping  
    dist :  43.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [783, 854, 865, 873]
    value i= 1 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813]]
    candidate_row : [[84, 855, 182, 876], [783, 854, 865, 873]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    bottom_row  [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    len of rows top =3 ,botttom 3 , i= 1, j= 2
    top_bbox = [764, 791, 865, 811] 
    bottom_bbox = [1110, 854, 1192, 873]
    Not overlaping  
    (right most)adding to mergeList  box  [764, 791, 865, 811]
    value i= 2 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811]]
    candidate_row : [[84, 855, 182, 876], [783, 854, 865, 873]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    bottom_row  [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    len of rows top =3 ,botttom 3 , i= 2, j= 2
    top_bbox = [1091, 791, 1192, 811] 
    bottom_bbox = [1110, 854, 1192, 873]
    overlaping  
    dist :  43.0
    distance not statisfy:  2
    adding to candidate row (bottom_bbox)  [1110, 854, 1192, 873]
    value i= 2 and j =3 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811]]
    candidate_row : [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  2
    value i= 3 and j =3 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811]]
    candidate_row : [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    candidate_row  [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    
    ---------------------------------------------------------------
    
    loop  12
    top_row  [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    bottom_row  [[878, 954, 960, 984], [1104, 959, 1192, 978]]
    len of rows top =3 ,botttom 2 , i= 0, j= 0
    top_bbox = [84, 855, 182, 876] 
    bottom_bbox = [878, 954, 960, 984]
    Not overlaping  
    (right most)adding to mergeList  box  [84, 855, 182, 876]
    value i= 1 and j =0 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    bottom_row  [[878, 954, 960, 984], [1104, 959, 1192, 978]]
    len of rows top =3 ,botttom 2 , i= 1, j= 0
    top_bbox = [783, 854, 865, 873] 
    bottom_bbox = [878, 954, 960, 984]
    Not overlaping  
    (right most)adding to mergeList  box  [783, 854, 865, 873]
    value i= 2 and j =0 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    bottom_row  [[878, 954, 960, 984], [1104, 959, 1192, 978]]
    len of rows top =3 ,botttom 2 , i= 2, j= 0
    top_bbox = [1110, 854, 1192, 873] 
    bottom_bbox = [878, 954, 960, 984]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [878, 954, 960, 984]
    value i= 2 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873]]
    candidate_row : [[878, 954, 960, 984]]
    
    --exsiting the while 1 loop--
    
    top_row  [[84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    bottom_row  [[878, 954, 960, 984], [1104, 959, 1192, 978]]
    len of rows top =3 ,botttom 2 , i= 2, j= 1
    top_bbox = [1110, 854, 1192, 873] 
    bottom_bbox = [1104, 959, 1192, 978]
    overlaping  
    dist :  86.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [1104, 959, 1192, 978]
    value i= 2 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873]]
    candidate_row : [[878, 954, 960, 984], [1104, 959, 1192, 978]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  2
    value i= 3 and j =2 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    candidate_row : [[878, 954, 960, 984], [1104, 959, 1192, 978]]
    candidate_row  [[878, 954, 960, 984], [1104, 959, 1192, 978]]
    
    ---------------------------------------------------------------
    
    loop  13
    top_row  [[878, 954, 960, 984], [1104, 959, 1192, 978]]
    bottom_row  [[926, 993, 961, 1022], [1149, 997, 1192, 1013]]
    len of rows top =2 ,botttom 2 , i= 0, j= 0
    top_bbox = [878, 954, 960, 984] 
    bottom_bbox = [926, 993, 961, 1022]
    overlaping  
    dist :  9.0
    distance statisfy :  0
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[878, 954, 961, 1022], [1104, 959, 1192, 978]]
    bottom_row  [[926, 993, 961, 1022], [1149, 997, 1192, 1013]]
    len of rows top =2 ,botttom 2 , i= 0, j= 1
    top_bbox = [878, 954, 961, 1022] 
    bottom_bbox = [1149, 997, 1192, 1013]
    Not overlaping  
    (right most) adding to candidate row (top box)  [878, 954, 961, 1022]
    (right most)adding to mergeList  box  [878, 954, 961, 1022]
    value i= 1 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022]]
    candidate_row : [[878, 954, 961, 1022]]
    
    --exsiting the while 1 loop--
    
    top_row  [[878, 954, 961, 1022], [1104, 959, 1192, 978]]
    bottom_row  [[926, 993, 961, 1022], [1149, 997, 1192, 1013]]
    len of rows top =2 ,botttom 2 , i= 1, j= 1
    top_bbox = [1104, 959, 1192, 978] 
    bottom_bbox = [1149, 997, 1192, 1013]
    overlaping  
    dist :  19.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [1149, 997, 1192, 1013]
    value i= 1 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022]]
    candidate_row : [[878, 954, 961, 1022], [1149, 997, 1192, 1013]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =2 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978]]
    candidate_row : [[878, 954, 961, 1022], [1149, 997, 1192, 1013]]
    candidate_row  [[878, 954, 961, 1022], [1149, 997, 1192, 1013]]
    
    ---------------------------------------------------------------
    
    loop  14
    top_row  [[878, 954, 961, 1022], [1149, 997, 1192, 1013]]
    bottom_row  [[913, 1055, 960, 1085], [1104, 1060, 1192, 1079]]
    len of rows top =2 ,botttom 2 , i= 0, j= 0
    top_bbox = [878, 954, 961, 1022] 
    bottom_bbox = [913, 1055, 960, 1085]
    overlaping  
    dist :  33.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [913, 1055, 960, 1085]
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978]]
    candidate_row : [[913, 1055, 960, 1085]]
    
    --exsiting the while 1 loop--
    
    top_row  [[878, 954, 961, 1022], [1149, 997, 1192, 1013]]
    bottom_row  [[913, 1055, 960, 1085], [1104, 1060, 1192, 1079]]
    len of rows top =2 ,botttom 2 , i= 0, j= 1
    top_bbox = [878, 954, 961, 1022] 
    bottom_bbox = [1104, 1060, 1192, 1079]
    Not overlaping  
    (right most)adding to mergeList  box  [878, 954, 961, 1022]
    value i= 1 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022]]
    candidate_row : [[913, 1055, 960, 1085]]
    
    --exsiting the while 1 loop--
    
    top_row  [[878, 954, 961, 1022], [1149, 997, 1192, 1013]]
    bottom_row  [[913, 1055, 960, 1085], [1104, 1060, 1192, 1079]]
    len of rows top =2 ,botttom 2 , i= 1, j= 1
    top_bbox = [1149, 997, 1192, 1013] 
    bottom_bbox = [1104, 1060, 1192, 1079]
    overlaping  
    dist :  47.0
    distance not statisfy:  1
    adding to candidate row (bottom_bbox)  [1104, 1060, 1192, 1079]
    value i= 1 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022]]
    candidate_row : [[913, 1055, 960, 1085], [1104, 1060, 1192, 1079]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =2 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022], [1149, 997, 1192, 1013]]
    candidate_row : [[913, 1055, 960, 1085], [1104, 1060, 1192, 1079]]
    candidate_row  [[913, 1055, 960, 1085], [1104, 1060, 1192, 1079]]
    
    ---------------------------------------------------------------
    
    loop  15
    top_row  [[913, 1055, 960, 1085], [1104, 1060, 1192, 1079]]
    bottom_row  [[830, 1089, 960, 1105], [1149, 1089, 1192, 1105]]
    len of rows top =2 ,botttom 2 , i= 0, j= 0
    top_bbox = [913, 1055, 960, 1085] 
    bottom_bbox = [830, 1089, 960, 1105]
    overlaping  
    dist :  4.0
    distance statisfy :  0
    value i= 0 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022], [1149, 997, 1192, 1013]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[830, 1055, 960, 1105], [1104, 1060, 1192, 1079]]
    bottom_row  [[830, 1089, 960, 1105], [1149, 1089, 1192, 1105]]
    len of rows top =2 ,botttom 2 , i= 0, j= 1
    top_bbox = [830, 1055, 960, 1105] 
    bottom_bbox = [1149, 1089, 1192, 1105]
    Not overlaping  
    (right most) adding to candidate row (top box)  [830, 1055, 960, 1105]
    (right most)adding to mergeList  box  [830, 1055, 960, 1105]
    value i= 1 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022], [1149, 997, 1192, 1013], [830, 1055, 960, 1105]]
    candidate_row : [[830, 1055, 960, 1105]]
    
    --exsiting the while 1 loop--
    
    top_row  [[830, 1055, 960, 1105], [1104, 1060, 1192, 1079]]
    bottom_row  [[830, 1089, 960, 1105], [1149, 1089, 1192, 1105]]
    len of rows top =2 ,botttom 2 , i= 1, j= 1
    top_bbox = [1104, 1060, 1192, 1079] 
    bottom_bbox = [1149, 1089, 1192, 1105]
    overlaping  
    dist :  10.0
    distance statisfy :  1
    adding *last* candidate row (top box)   [1104, 1060, 1192, 1105]
    value i= 1 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022], [1149, 997, 1192, 1013], [830, 1055, 960, 1105]]
    candidate_row : [[830, 1055, 960, 1105], [1104, 1060, 1192, 1105]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =2 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022], [1149, 997, 1192, 1013], [830, 1055, 960, 1105], [1104, 1060, 1192, 1105]]
    candidate_row : [[830, 1055, 960, 1105], [1104, 1060, 1192, 1105]]
    candidate_row  [[830, 1055, 960, 1105], [1104, 1060, 1192, 1105]]
    
    ---------------------------------------------------------------
    
    loop  16
    top_row  [[830, 1055, 960, 1105], [1104, 1060, 1192, 1105]]
    bottom_row  [[1091, 1151, 1192, 1171], [764, 1152, 960, 1173]]
    len of rows top =2 ,botttom 2 , i= 0, j= 0
    top_bbox = [830, 1055, 960, 1105] 
    bottom_bbox = [1091, 1151, 1192, 1171]
    Not overlaping  
    (right most)adding to mergeList  box  [830, 1055, 960, 1105]
    value i= 1 and j =0 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022], [1149, 997, 1192, 1013], [830, 1055, 960, 1105], [1104, 1060, 1192, 1105], [830, 1055, 960, 1105]]
    candidate_row : []
    
    --exsiting the while 1 loop--
    
    top_row  [[830, 1055, 960, 1105], [1104, 1060, 1192, 1105]]
    bottom_row  [[1091, 1151, 1192, 1171], [764, 1152, 960, 1173]]
    len of rows top =2 ,botttom 2 , i= 1, j= 0
    top_bbox = [1104, 1060, 1192, 1105] 
    bottom_bbox = [1091, 1151, 1192, 1171]
    overlaping  
    dist :  46.0
    distance not statisfy:  0
    adding to candidate row (bottom_bbox)  [1091, 1151, 1192, 1171]
    value i= 1 and j =1 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022], [1149, 997, 1192, 1013], [830, 1055, 960, 1105], [1104, 1060, 1192, 1105], [830, 1055, 960, 1105]]
    candidate_row : [[1091, 1151, 1192, 1171]]
    
    --exsiting the while 1 loop--
    
    top_row  [[830, 1055, 960, 1105], [1104, 1060, 1192, 1105]]
    bottom_row  [[1091, 1151, 1192, 1171], [764, 1152, 960, 1173]]
    len of rows top =2 ,botttom 2 , i= 1, j= 1
    top_bbox = [1104, 1060, 1192, 1105] 
    bottom_bbox = [764, 1152, 960, 1173]
    Not overlaping  
    (left most) adding to candidate row (bottom_bbox)  [764, 1152, 960, 1173]
    value i= 1 and j =2 
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022], [1149, 997, 1192, 1013], [830, 1055, 960, 1105], [1104, 1060, 1192, 1105], [830, 1055, 960, 1105]]
    candidate_row : [[1091, 1151, 1192, 1171], [764, 1152, 960, 1173]]
    
    --exsiting the while 1 loop--
    
    top box  is not empty :  1
    value i= 2 and j =2 
    exsiting the while 2 loop
    merge_list :  [[927, 88, 985, 104], [1034, 88, 1192, 136], [100, 116, 176, 268], [858, 117, 985, 133], [1034, 88, 1192, 136], [1107, 147, 1192, 163], [1113, 175, 1193, 192], [84, 422, 136, 438], [405, 422, 450, 438], [485, 422, 536, 438], [649, 422, 807, 438], [932, 422, 1191, 506], [82, 450, 245, 525], [404, 450, 516, 467], [648, 451, 736, 467], [1080, 453, 1122, 495], [82, 450, 245, 525], [405, 521, 498, 537], [82, 538, 273, 584], [404, 549, 516, 566], [82, 538, 273, 584], [84, 730, 194, 751], [819, 726, 865, 755], [994, 730, 1029, 751], [1096, 730, 1191, 746], [84, 791, 153, 813], [764, 791, 865, 811], [1091, 791, 1192, 811], [84, 855, 182, 876], [783, 854, 865, 873], [1110, 854, 1192, 873], [878, 954, 961, 1022], [1104, 959, 1192, 978], [878, 954, 961, 1022], [1149, 997, 1192, 1013], [830, 1055, 960, 1105], [1104, 1060, 1192, 1105], [830, 1055, 960, 1105], [1104, 1060, 1192, 1105]]
    candidate_row : [[1091, 1151, 1192, 1171], [764, 1152, 960, 1173]]
    candidate_row  [[1091, 1151, 1192, 1171], [764, 1152, 960, 1173]]
    
    ---------------------------------------------------------------
    
    




    -1




```python
def display_bounding_box(img, word_bbox_list,title="word bounding box"):
    for bbox in word_bbox_list: 
        if len(bbox)==3:
            b = bbox[-1]
#             print(bbox)
        else:
            b=bbox
        img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

    # show annotated image and wait for keypress
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
```


```python
p="C:/Users/Sanjeev/Downloads/software/0325updated.task2train(626p)-20200414T125907Z-001/0325updated.task2train(626p)/X00016469612.jpg"
def print_ocr_string(img):  
    res = isinstance(img, str)
    if res:
        print('reading image ../n' )
        img = cv2.imread(img)  
    text = pytesseract.image_to_string(img)
#     print(text)
    return text
```


```python
print_ocr_string(p)
```

    reading image ../n
    




    'tan woon yann\n\nBOOK TAK (TAMAN DAYA) SDN BHD\nB97\nNO.5? 55,57 & 59, JALAN SAGU 18,\nTAMAN DAYA\n81100 JOHOR BAHRU,\nJOHOR.\n\nWAM MICA A\n\nDocument Ho : TDO1167104\n\n \n\nDate 25/12/2018 8:13:39 PM\nCashier MANIS\nMember\nCASH BILL\nCODE/DESC PRICE Disc AMOUITT\nQuy RM RM\n9556939040118 KF MODELLING CLAY KIDDY FISH\n1PC * 9.00) 6,00 9.00\nTotal : 9,00\nRour ding Adjustment 0.00\n\nRound::d Total (RM):\n\n9.60\n\nCash\nCHANGE\n\n  \n\nGOODS SOLD ARE NOT RETURNAR\nEXCHANGEABLE\n\n \n\nTHANK YOU.\nPLEASE COME AGA t'




```python
pip list -v
```

    Package                            Version     Location                          Installer
    ---------------------------------- ----------- --------------------------------- ---------
    -cipy                              1.3.1       c:\installation\lib\site-packages          
    absl-py                            0.9.0       c:\installation\lib\site-packages pip      
    alabaster                          0.7.12      c:\installation\lib\site-packages pip      
    anaconda-client                    1.7.2       c:\installation\lib\site-packages          
    anaconda-navigator                 1.9.7       c:\installation\lib\site-packages          
    anaconda-project                   0.8.3       c:\installation\lib\site-packages conda    
    asn1crypto                         1.0.1       c:\installation\lib\site-packages conda    
    astor                              0.8.1       c:\installation\lib\site-packages pip      
    astroid                            2.3.1       c:\installation\lib\site-packages conda    
    astropy                            3.2.1       c:\installation\lib\site-packages          
    atomicwrites                       1.3.0       c:\installation\lib\site-packages          
    attrs                              19.2.0      c:\installation\lib\site-packages conda    
    Babel                              2.7.0       c:\installation\lib\site-packages pip      
    backcall                           0.1.0       c:\installation\lib\site-packages pip      
    backports.functools-lru-cache      1.5         c:\installation\lib\site-packages          
    backports.os                       0.1.1       c:\installation\lib\site-packages          
    backports.shutil-get-terminal-size 1.0.0       c:\installation\lib\site-packages pip      
    backports.tempfile                 1.0         c:\installation\lib\site-packages pip      
    backports.weakref                  1.0.post1   c:\installation\lib\site-packages          
    beautifulsoup4                     4.8.0       c:\installation\lib\site-packages pip      
    bitarray                           1.0.1       c:\installation\lib\site-packages pip      
    bkcharts                           0.2         c:\installation\lib\site-packages          
    bleach                             3.1.0       c:\installation\lib\site-packages          
    blis                               0.4.1       c:\installation\lib\site-packages conda    
    bokeh                              1.3.4       c:\installation\lib\site-packages pip      
    boto                               2.49.0      c:\installation\lib\site-packages          
    boto3                              1.12.21     c:\installation\lib\site-packages pip      
    botocore                           1.15.21     c:\installation\lib\site-packages pip      
    Bottleneck                         1.2.1       c:\installation\lib\site-packages          
    bpemb                              0.3.0       c:\installation\lib\site-packages pip      
    cachetools                         4.0.0       c:\installation\lib\site-packages pip      
    catalogue                          1.0.0       c:\installation\lib\site-packages conda    
    certifi                            2019.9.11   c:\installation\lib\site-packages          
    cffi                               1.12.3      c:\installation\lib\site-packages pip      
    chardet                            3.0.4       c:\installation\lib\site-packages conda    
    chart-studio                       1.0.0       c:\installation\lib\site-packages pip      
    Click                              7.0         c:\installation\lib\site-packages          
    click-plugins                      1.1.1       c:\installation\lib\site-packages pip      
    cligj                              0.5.0       c:\installation\lib\site-packages          
    cloudpickle                        1.2.2       c:\installation\lib\site-packages conda    
    clyent                             1.2.2       c:\installation\lib\site-packages          
    colorama                           0.4.1       c:\installation\lib\site-packages          
    comtypes                           1.1.7       c:\installation\lib\site-packages          
    conda                              4.8.3       c:\installation\lib\site-packages          
    conda-build                        3.18.9      c:\installation\lib\site-packages          
    conda-package-handling             1.6.0       c:\installation\lib\site-packages conda    
    conda-verify                       3.4.2       c:\installation\lib\site-packages conda    
    contextlib2                        0.6.0       c:\installation\lib\site-packages conda    
    cryptography                       2.7         c:\installation\lib\site-packages pip      
    cycler                             0.10.0      c:\installation\lib\site-packages          
    cymem                              2.0.3       c:\installation\lib\site-packages conda    
    Cython                             0.29.13     c:\installation\lib\site-packages pip      
    cytoolz                            0.10.0      c:\installation\lib\site-packages          
    dask                               2.5.2       c:\installation\lib\site-packages conda    
    decorator                          4.4.0       c:\installation\lib\site-packages pip      
    defusedxml                         0.6.0       c:\installation\lib\site-packages          
    descartes                          1.1.0       c:\installation\lib\site-packages conda    
    distributed                        2.5.2       c:\installation\lib\site-packages conda    
    docutils                           0.15.2      c:\installation\lib\site-packages pip      
    en-core-web-sm                     2.2.5       c:\installation\lib\site-packages pip      
    entrypoints                        0.3         c:\installation\lib\site-packages          
    et-xmlfile                         1.0.1       c:\installation\lib\site-packages          
    fastcache                          1.1.0       c:\installation\lib\site-packages pip      
    filelock                           3.0.12      c:\installation\lib\site-packages pip      
    Fiona                              1.8.4       c:\installation\lib\site-packages          
    Flask                              1.1.1       c:\installation\lib\site-packages conda    
    fsspec                             0.5.2       c:\installation\lib\site-packages conda    
    future                             0.17.1      c:\installation\lib\site-packages          
    gast                               0.2.2       c:\installation\lib\site-packages pip      
    GDAL                               2.3.3       c:\installation\lib\site-packages          
    gensim                             3.8.1       c:\installation\lib\site-packages pip      
    geojson                            2.5.0       c:\installation\lib\site-packages pip      
    geopandas                          0.6.1       c:\installation\lib\site-packages conda    
    gevent                             1.4.0       c:\installation\lib\site-packages          
    glob2                              0.7         c:\installation\lib\site-packages conda    
    google-auth                        1.11.3      c:\installation\lib\site-packages pip      
    google-auth-oauthlib               0.4.1       c:\installation\lib\site-packages pip      
    google-pasta                       0.2.0       c:\installation\lib\site-packages pip      
    googledrivedownloader              0.4         c:\installation\lib\site-packages pip      
    greenlet                           0.4.15      c:\installation\lib\site-packages pip      
    grpcio                             1.27.2      c:\installation\lib\site-packages pip      
    h5py                               2.9.0       c:\installation\lib\site-packages pip      
    HeapDict                           1.0.1       c:\installation\lib\site-packages conda    
    html5lib                           1.0.1       c:\installation\lib\site-packages pip      
    idna                               2.8         c:\installation\lib\site-packages          
    idx2numpy                          1.2.2       c:\installation\lib\site-packages pip      
    imageio                            2.6.0       c:\installation\lib\site-packages conda    
    imagesize                          1.1.0       c:\installation\lib\site-packages pip      
    importlib-metadata                 0.23        c:\installation\lib\site-packages conda    
    ipykernel                          5.1.2       c:\installation\lib\site-packages pip      
    ipython                            7.8.0       c:\installation\lib\site-packages pip      
    ipython-genutils                   0.2.0       c:\installation\lib\site-packages          
    ipywidgets                         7.5.1       c:\installation\lib\site-packages conda    
    isodate                            0.6.0       c:\installation\lib\site-packages pip      
    isort                              4.3.21      c:\installation\lib\site-packages pip      
    itsdangerous                       1.1.0       c:\installation\lib\site-packages pip      
    jdcal                              1.4.1       c:\installation\lib\site-packages pip      
    jedi                               0.15.1      c:\installation\lib\site-packages pip      
    Jinja2                             2.10.3      c:\installation\lib\site-packages conda    
    jmespath                           0.9.5       c:\installation\lib\site-packages pip      
    joblib                             0.13.2      c:\installation\lib\site-packages          
    json5                              0.8.5       c:\installation\lib\site-packages conda    
    jsonschema                         3.0.2       c:\installation\lib\site-packages pip      
    jupyter                            1.0.0       c:\installation\lib\site-packages pip      
    jupyter-client                     5.3.3       c:\installation\lib\site-packages conda    
    jupyter-console                    6.0.0       c:\installation\lib\site-packages          
    jupyter-core                       4.5.0       c:\installation\lib\site-packages conda    
    jupyterlab                         1.1.4       c:\installation\lib\site-packages          
    jupyterlab-server                  1.0.6       c:\installation\lib\site-packages conda    
    Keras                              2.3.1       c:\installation\lib\site-packages pip      
    Keras-Applications                 1.0.8       c:\installation\lib\site-packages pip      
    Keras-Preprocessing                1.1.0       c:\installation\lib\site-packages pip      
    keyring                            18.0.0      c:\installation\lib\site-packages pip      
    kiwisolver                         1.1.0       c:\installation\lib\site-packages pip      
    lazy-object-proxy                  1.4.2       c:\installation\lib\site-packages pip      
    libarchive-c                       2.8         c:\installation\lib\site-packages          
    llvmlite                           0.29.0      c:\installation\lib\site-packages          
    locket                             0.2.0       c:\installation\lib\site-packages          
    lxml                               4.4.1       c:\installation\lib\site-packages pip      
    Markdown                           3.2.1       c:\installation\lib\site-packages pip      
    MarkupSafe                         1.1.1       c:\installation\lib\site-packages pip      
    matplotlib                         3.1.3       c:\installation\lib\site-packages pip      
    mccabe                             0.6.1       c:\installation\lib\site-packages          
    menuinst                           1.4.16      c:\installation\lib\site-packages          
    mistune                            0.8.4       c:\installation\lib\site-packages pip      
    mkl-fft                            1.0.14      c:\installation\lib\site-packages          
    mkl-random                         1.1.0       c:\installation\lib\site-packages pip      
    mkl-service                        2.3.0       c:\installation\lib\site-packages          
    mock                               3.0.5       c:\installation\lib\site-packages pip      
    more-itertools                     7.2.0       c:\installation\lib\site-packages conda    
    mpmath                             1.1.0       c:\installation\lib\site-packages          
    msgpack                            0.6.1       c:\installation\lib\site-packages pip      
    multipledispatch                   0.6.0       c:\installation\lib\site-packages          
    munch                              2.5.0       c:\installation\lib\site-packages conda    
    murmurhash                         1.0.0       c:\installation\lib\site-packages          
    navigator-updater                  0.2.1       c:\installation\lib\site-packages          
    nbconvert                          5.6.0       c:\installation\lib\site-packages conda    
    nbformat                           4.4.0       c:\installation\lib\site-packages pip      
    networkx                           2.3         c:\installation\lib\site-packages pip      
    nltk                               3.4.5       c:\installation\lib\site-packages pip      
    nose                               1.3.7       c:\installation\lib\site-packages          
    notebook                           6.0.1       c:\installation\lib\site-packages conda    
    numba                              0.45.1      c:\installation\lib\site-packages          
    numexpr                            2.7.0       c:\installation\lib\site-packages          
    numpy                              1.16.5      c:\installation\lib\site-packages          
    numpydoc                           0.9.1       c:\installation\lib\site-packages pip      
    oauthlib                           3.1.0       c:\installation\lib\site-packages pip      
    olefile                            0.46        c:\installation\lib\site-packages          
    opencv-python                      4.2.0.32    c:\installation\lib\site-packages pip      
    openpyxl                           3.0.0       c:\installation\lib\site-packages conda    
    opt-einsum                         3.2.0       c:\installation\lib\site-packages pip      
    packaging                          19.2        c:\installation\lib\site-packages conda    
    pandas                             0.25.1      c:\installation\lib\site-packages pip      
    pandocfilters                      1.4.2       c:\installation\lib\site-packages pip      
    parso                              0.5.1       c:\installation\lib\site-packages conda    
    partd                              1.0.0       c:\installation\lib\site-packages conda    
    path.py                            12.0.1      c:\installation\lib\site-packages pip      
    pathlib2                           2.3.5       c:\installation\lib\site-packages conda    
    patsy                              0.5.1       c:\installation\lib\site-packages          
    pdf2image                          1.12.1      c:\installation\lib\site-packages pip      
    pep8                               1.7.1       c:\installation\lib\site-packages          
    pickleshare                        0.7.5       c:\installation\lib\site-packages          
    Pillow                             6.2.0       c:\installation\lib\site-packages conda    
    pip                                19.2.3      c:\installation\lib\site-packages          
    pkginfo                            1.5.0.1     c:\installation\lib\site-packages          
    plac                               0.9.6       c:\installation\lib\site-packages          
    plotly                             4.5.4       c:\installation\lib\site-packages pip      
    pluggy                             0.13.0      c:\installation\lib\site-packages conda    
    ply                                3.11        c:\installation\lib\site-packages          
    plyfile                            0.7.2       c:\installation\lib\site-packages pip      
    preshed                            3.0.2       c:\installation\lib\site-packages conda    
    prometheus-client                  0.7.1       c:\installation\lib\site-packages conda    
    prompt-toolkit                     2.0.10      c:\installation\lib\site-packages conda    
    protobuf                           3.11.3      c:\installation\lib\site-packages pip      
    psutil                             5.6.3       c:\installation\lib\site-packages pip      
    py                                 1.8.0       c:\installation\lib\site-packages pip      
    pyasn1                             0.4.8       c:\installation\lib\site-packages pip      
    pyasn1-modules                     0.2.8       c:\installation\lib\site-packages pip      
    pycodestyle                        2.5.0       c:\installation\lib\site-packages          
    pycosat                            0.6.3       c:\installation\lib\site-packages          
    pycparser                          2.19        c:\installation\lib\site-packages          
    pycrypto                           2.6.1       c:\installation\lib\site-packages          
    pycurl                             7.43.0.3    c:\installation\lib\site-packages          
    pyflakes                           2.1.1       c:\installation\lib\site-packages pip      
    Pygments                           2.4.2       c:\installation\lib\site-packages pip      
    pylint                             2.4.2       c:\installation\lib\site-packages conda    
    pyodbc                             4.0.27      c:\installation\lib\site-packages pip      
    pyOpenSSL                          19.0.0      c:\installation\lib\site-packages          
    pyparsing                          2.4.2       c:\installation\lib\site-packages conda    
    PyPDF4                             1.27.0      c:\installation\lib\site-packages pip      
    pyproj                             1.9.6       c:\installation\lib\site-packages          
    pyreadline                         2.1         c:\installation\lib\site-packages          
    pyrsistent                         0.15.4      c:\installation\lib\site-packages conda    
    pyshp                              2.1.0       c:\installation\lib\site-packages pip      
    PySocks                            1.7.1       c:\installation\lib\site-packages conda    
    pytesseract                        0.3.3       c:\installation\lib\site-packages pip      
    pytest                             5.2.1       c:\installation\lib\site-packages          
    pytest-arraydiff                   0.3         c:\installation\lib\site-packages pip      
    pytest-astropy                     0.5.0       c:\installation\lib\site-packages pip      
    pytest-doctestplus                 0.4.0       c:\installation\lib\site-packages conda    
    pytest-openfiles                   0.4.0       c:\installation\lib\site-packages conda    
    pytest-remotedata                  0.3.2       c:\installation\lib\site-packages conda    
    python-dateutil                    2.8.0       c:\installation\lib\site-packages pip      
    python-mnist                       0.6         c:\installation\lib\site-packages pip      
    pytorch-nlp                        0.5.0       c:\installation\lib\site-packages pip      
    pytorch-transformers               1.2.0       c:\installation\lib\site-packages pip      
    pytz                               2019.3      c:\installation\lib\site-packages conda    
    PyWavelets                         1.0.3       c:\installation\lib\site-packages pip      
    pywin32                            223         c:\installation\lib\site-packages          
    pywinpty                           0.5.5       c:\installation\lib\site-packages          
    PyYAML                             5.1.2       c:\installation\lib\site-packages          
    pyzmq                              18.1.0      c:\installation\lib\site-packages          
    QtAwesome                          0.6.0       c:\installation\lib\site-packages conda    
    qtconsole                          4.5.5       c:\installation\lib\site-packages conda    
    QtPy                               1.9.0       c:\installation\lib\site-packages conda    
    rdflib                             4.2.2       c:\installation\lib\site-packages pip      
    regex                              2020.4.4    c:\installation\lib\site-packages pip      
    requests                           2.22.0      c:\installation\lib\site-packages pip      
    requests-oauthlib                  1.3.0       c:\installation\lib\site-packages pip      
    retrying                           1.3.3       c:\installation\lib\site-packages pip      
    rope                               0.14.0      c:\installation\lib\site-packages          
    rope-py3k                          0.9.4.post1 c:\installation\lib\site-packages pip      
    rsa                                4.0         c:\installation\lib\site-packages pip      
    Rtree                              0.9.3       c:\installation\lib\site-packages pip      
    ruamel-yaml                        0.15.46     c:\installation\lib\site-packages          
    s3transfer                         0.3.3       c:\installation\lib\site-packages pip      
    sacremoses                         0.0.38      c:\installation\lib\site-packages pip      
    scikit-image                       0.15.0      c:\installation\lib\site-packages pip      
    scikit-learn                       0.21.3      c:\installation\lib\site-packages pip      
    scipy                              1.4.1       c:\installation\lib\site-packages pip      
    seaborn                            0.9.0       c:\installation\lib\site-packages pip      
    Send2Trash                         1.5.0       c:\installation\lib\site-packages          
    sentencepiece                      0.1.85      c:\installation\lib\site-packages pip      
    setuptools                         41.4.0      c:\installation\lib\site-packages          
    Shapely                            1.6.4.post1 c:\installation\lib\site-packages          
    simplegeneric                      0.8.1       c:\installation\lib\site-packages          
    singledispatch                     3.4.0.3     c:\installation\lib\site-packages          
    six                                1.12.0      c:\installation\lib\site-packages          
    smart-open                         1.9.0       c:\installation\lib\site-packages pip      
    snowballstemmer                    2.0.0       c:\installation\lib\site-packages conda    
    sortedcollections                  1.1.2       c:\installation\lib\site-packages pip      
    sortedcontainers                   2.1.0       c:\installation\lib\site-packages          
    soupsieve                          1.9.3       c:\installation\lib\site-packages conda    
    spacy                              2.2.3       c:\installation\lib\site-packages conda    
    spacy-lookups-data                 0.2.0       c:\installation\lib\site-packages conda    
    Sphinx                             2.2.0       c:\installation\lib\site-packages conda    
    sphinxcontrib-applehelp            1.0.1       c:\installation\lib\site-packages pip      
    sphinxcontrib-devhelp              1.0.1       c:\installation\lib\site-packages pip      
    sphinxcontrib-htmlhelp             1.0.2       c:\installation\lib\site-packages pip      
    sphinxcontrib-jsmath               1.0.1       c:\installation\lib\site-packages pip      
    sphinxcontrib-qthelp               1.0.2       c:\installation\lib\site-packages pip      
    sphinxcontrib-serializinghtml      1.1.3       c:\installation\lib\site-packages pip      
    sphinxcontrib-websupport           1.1.2       c:\installation\lib\site-packages pip      
    spyder                             3.3.6       c:\installation\lib\site-packages pip      
    spyder-kernels                     0.5.2       c:\installation\lib\site-packages conda    
    SQLAlchemy                         1.3.9       c:\installation\lib\site-packages conda    
    srsly                              1.0.0       c:\installation\lib\site-packages conda    
    statsmodels                        0.10.1      c:\installation\lib\site-packages          
    sympy                              1.4         c:\installation\lib\site-packages          
    tables                             3.5.2       c:\installation\lib\site-packages          
    tblib                              1.4.0       c:\installation\lib\site-packages pip      
    tensorboard                        2.0.2       c:\installation\lib\site-packages pip      
    tensorflow                         2.0.0       c:\installation\lib\site-packages pip      
    tensorflow-estimator               2.0.1       c:\installation\lib\site-packages pip      
    termcolor                          1.1.0       c:\installation\lib\site-packages pip      
    terminado                          0.8.2       c:\installation\lib\site-packages          
    testpath                           0.4.2       c:\installation\lib\site-packages pip      
    thinc                              7.3.0       c:\installation\lib\site-packages conda    
    toolz                              0.10.0      c:\installation\lib\site-packages conda    
    torch                              1.4.0       c:\installation\lib\site-packages          
    torch-cluster                      1.5.3       c:\installation\lib\site-packages pip      
    torch-geometric                    1.4.3       c:\installation\lib\site-packages pip      
    torch-scatter                      2.0.4       c:\installation\lib\site-packages pip      
    torch-sparse                       0.6.1       c:\installation\lib\site-packages pip      
    torch-spline-conv                  1.2.0       c:\installation\lib\site-packages pip      
    torchvision                        0.5.0       c:\installation\lib\site-packages          
    tornado                            6.0.3       c:\installation\lib\site-packages pip      
    tqdm                               4.36.1      c:\installation\lib\site-packages conda    
    traitlets                          4.3.3       c:\installation\lib\site-packages conda    
    unicodecsv                         0.14.1      c:\installation\lib\site-packages          
    urllib3                            1.24.2      c:\installation\lib\site-packages          
    wasabi                             0.6.0       c:\installation\lib\site-packages conda    
    wcwidth                            0.1.7       c:\installation\lib\site-packages          
    webencodings                       0.5.1       c:\installation\lib\site-packages          
    Werkzeug                           0.16.0      c:\installation\lib\site-packages conda    
    wheel                              0.33.6      c:\installation\lib\site-packages          
    widgetsnbextension                 3.5.1       c:\installation\lib\site-packages pip      
    win-inet-pton                      1.1.0       c:\installation\lib\site-packages          
    win-unicode-console                0.5         c:\installation\lib\site-packages          
    wincertstore                       0.2         c:\installation\lib\site-packages          
    wrapt                              1.11.2      c:\installation\lib\site-packages pip      
    xlrd                               1.2.0       c:\installation\lib\site-packages pip      
    XlsxWriter                         1.2.1       c:\installation\lib\site-packages conda    
    xlwings                            0.15.10     c:\installation\lib\site-packages          
    xlwt                               1.3.0       c:\installation\lib\site-packages          
    zict                               1.0.0       c:\installation\lib\site-packages          
    zipp                               0.6.0       c:\installation\lib\site-packages conda    
    1 location(s) to search for versions of pip:
    * https://pypi.org/simple/pip/
    Getting page https://pypi.org/simple/pip/
    Found index url https://pypi.org/simple
    Getting credentials from keyring for https://pypi.org/simple
    Getting credentials from keyring for pypi.org
    Looking up "https://pypi.org/simple/pip/" in the cache
    Request header has "max_age" as 0, cache bypassed
    Starting new HTTPS connection (1): pypi.org:443
    https://pypi.org:443 "GET /simple/pip/ HTTP/1.1" 200 13051
    Updating cache with response from "https://pypi.org/simple/pip/"
    Caching due to etag
    Analyzing links from page https://pypi.org/simple/pip/
      Found link https://files.pythonhosted.org/packages/3d/9d/1e313763bdfb6a48977b65829c6ce2a43eaae29ea2f907c8bbef024a7219/pip-0.2.tar.gz#sha256=88bb8d029e1bf4acd0e04d300104b7440086f94cc1ce1c5c3c31e3293aee1f81 (from https://pypi.org/simple/pip/), version: 0.2
      Found link https://files.pythonhosted.org/packages/18/ad/c0fe6cdfe1643a19ef027c7168572dac6283b80a384ddf21b75b921877da/pip-0.2.1.tar.gz#sha256=83522005c1266cc2de97e65072ff7554ac0f30ad369c3b02ff3a764b962048da (from https://pypi.org/simple/pip/), version: 0.2.1
      Found link https://files.pythonhosted.org/packages/17/05/f66144ef69b436d07f8eeeb28b7f77137f80de4bf60349ec6f0f9509e801/pip-0.3.tar.gz#sha256=183c72455cb7f8860ac1376f8c4f14d7f545aeab8ee7c22cd4caf79f35a2ed47 (from https://pypi.org/simple/pip/), version: 0.3
      Found link https://files.pythonhosted.org/packages/0a/bb/d087c9a1415f8726e683791c0b2943c53f2b76e69f527f2e2b2e9f9e7b5c/pip-0.3.1.tar.gz#sha256=34ce534f17065c78f980702928e988a6b6b2d8a9851aae5f1571a1feb9bb58d8 (from https://pypi.org/simple/pip/), version: 0.3.1
      Found link https://files.pythonhosted.org/packages/cf/c3/153571aaac6cf999f4bb09c019b1ff379b7b599ea833813a41c784eec995/pip-0.4.tar.gz#sha256=28fc67558874f71fddda7168f73595f1650523dce3bc5bf189713ecdfc1e456e (from https://pypi.org/simple/pip/), version: 0.4
      Found link https://files.pythonhosted.org/packages/8d/c7/f05c87812fa5d9562ecbc5f4f1fc1570444f53c81c834a7f662af406e3c1/pip-0.5.tar.gz#sha256=328d8412782f22568508a0d0c78a49c9920a82e44c8dfca49954fe525c152b2a (from https://pypi.org/simple/pip/), version: 0.5
      Found link https://files.pythonhosted.org/packages/9a/aa/f536b6d14fe03343367da2ff44eee28f340ae650cd017ca088b6be13084a/pip-0.5.1.tar.gz#sha256=e27650538c41fe1007a41abd4cfd0f905b822622cbe1f8e7e09d1215af207694 (from https://pypi.org/simple/pip/), version: 0.5.1
      Found link https://files.pythonhosted.org/packages/db/e6/fdf7be8a17b032c533d3f91e91e2c63dd81d3627cbe4113248a00c2d39d8/pip-0.6.tar.gz#sha256=4cf47db6815b2f435d1f44e1f35ff04823043f6161f7df9aec71a123b0c47f0d (from https://pypi.org/simple/pip/), version: 0.6
      Found link https://files.pythonhosted.org/packages/91/cd/105f4d3c75d0ae18e12623acc96f42168aaba408dd6e43c4505aa21f8e37/pip-0.6.1.tar.gz#sha256=efe47e84ffeb0ea4804f9858b8a94bebd07f5452f907ebed36d03aed06a9f9ec (from https://pypi.org/simple/pip/), version: 0.6.1
      Found link https://files.pythonhosted.org/packages/1c/c7/c0e1a9413c37828faf290f29a85a4d6034c145cc04bf1622ba8beb662ad8/pip-0.6.2.tar.gz#sha256=1c1a504d7e70d2c24246f95bd16e3d5fcec740fd144df69a407bf65a2ee67586 (from https://pypi.org/simple/pip/), version: 0.6.2
      Found link https://files.pythonhosted.org/packages/3f/af/c4b9d49fb0f286996b28dbc0955c3ad359794697eb98e0e69863908070b0/pip-0.6.3.tar.gz#sha256=1a6df71eb29b98cba11bde6d6a0d8c6dd8b0518e74ceb71fb31ea4fbb42fd313 (from https://pypi.org/simple/pip/), version: 0.6.3
      Found link https://files.pythonhosted.org/packages/ec/7a/6fe91ff0079ad0437830957c459d52f3923e516f5b453218f2a93d09a427/pip-0.7.tar.gz#sha256=ceaea0b9e494d893c8a191895301b79c1db33e41f14d3ad93e3d28a8b4e9bf27 (from https://pypi.org/simple/pip/), version: 0.7
      Found link https://files.pythonhosted.org/packages/a5/63/11303863c2f5e9d9a15d89fcf7513a4b60987007d418862e0fb65c09fff7/pip-0.7.1.tar.gz#sha256=f54f05aa17edd0036de433c44892c8fedb1fd2871c97829838feb995818d24c3 (from https://pypi.org/simple/pip/), version: 0.7.1
      Found link https://files.pythonhosted.org/packages/cd/a9/1debaa96bbc1005c1c8ad3b79fec58c198d35121546ea2e858ce0894268a/pip-0.7.2.tar.gz#sha256=98df2eb779358412bbbae75980171ae85deebc846d87e244d086520b1212da09 (from https://pypi.org/simple/pip/), version: 0.7.2
      Found link https://files.pythonhosted.org/packages/74/54/f785c327fb3d163560a879b36edae5c78ee07806be282c9d4807f6be7dd1/pip-0.8.tar.gz#sha256=9017e4484a212dd4e1a43dd9f039dd7fc8338d4eea1c339d5ae1c80726de5b0f (from https://pypi.org/simple/pip/), version: 0.8
      Found link https://files.pythonhosted.org/packages/5c/79/5e8381cc3078bae92166f2ba96de8355e8c181926505ba8882f7b099a500/pip-0.8.1.tar.gz#sha256=7176a87f35675f6468341212f3b959bb51d23ea66eb1c3692bf746c45c716fa2 (from https://pypi.org/simple/pip/), version: 0.8.1
      Found link https://files.pythonhosted.org/packages/17/3e/0a98ab032991518741e7e712a719633e6ae160f51b3d3e855194530fd308/pip-0.8.2.tar.gz#sha256=f80a3549c048bc3bbcb47844826e9c7c6fcd87e77b92bef0d9e66d1b397c4962 (from https://pypi.org/simple/pip/), version: 0.8.2
      Found link https://files.pythonhosted.org/packages/f7/9a/943fc6d879ed7220bac2e7e53096bfe78abec88d77f2f516400e0129679e/pip-0.8.3.tar.gz#sha256=1be2e18edd38aa75b5e4ef38a99ec33ba9247177cfcb4a6d2d2b3e73430e3001 (from https://pypi.org/simple/pip/), version: 0.8.3
      Found link https://files.pythonhosted.org/packages/24/33/6eb675fb6db7b71d69d6928b33dea61b8bf5cfe1e5649be70ec84ce2fc09/pip-1.0.tar.gz#sha256=34ba07e2d14ba86d5088ba896ac80bed845a9b276ab8acb279b8d99bc77fec8e (from https://pypi.org/simple/pip/), version: 1.0
      Found link https://files.pythonhosted.org/packages/10/d9/f584e6107ef98ad7eaaaa5d0f756bfee12561fa6a4712ffdb7209e0e1fd4/pip-1.0.1.tar.gz#sha256=37d2f18213d3845d2038dd3686bc71fc12bb41ad66c945a8b0dfec2879f3497b (from https://pypi.org/simple/pip/), version: 1.0.1
      Found link https://files.pythonhosted.org/packages/16/90/5e6f80364d8a656f60681dfb7330298edef292d43e1499bcb3a4c71ff0b9/pip-1.0.2.tar.gz#sha256=a6ed9b36aac2f121c01a2c9e0307a9e4d9438d100a407db701ac65479a3335d2 (from https://pypi.org/simple/pip/), version: 1.0.2
      Found link https://files.pythonhosted.org/packages/25/57/0d42cf5307d79913a082c5c4397d46f3793bc35e1138a694136d6e31be99/pip-1.1.tar.gz#sha256=993804bb947d18508acee02141281c77d27677f8c14eaa64d6287a1c53ef01c8 (from https://pypi.org/simple/pip/), version: 1.1
      Found link https://files.pythonhosted.org/packages/ba/c3/4e1f892f41aaa217fe0d1f827fa05928783349c69f3cc06fdd68e112678a/pip-1.2.tar.gz#sha256=2b168f1987403f1dc6996a1f22a6f6637b751b7ab6ff27e78380b8d6e70aa314 (from https://pypi.org/simple/pip/), version: 1.2
      Found link https://files.pythonhosted.org/packages/c3/a2/a63244da32afd9ce9a8ca1bd86e71610039adea8b8314046ebe5047527a6/pip-1.2.1.tar.gz#sha256=12a9302acfca62cdc7bc5d83386cac3e0581db61ac39acdb3a4e766a16b88eb1 (from https://pypi.org/simple/pip/), version: 1.2.1
      Found link https://files.pythonhosted.org/packages/00/45/69d4f2602b80550bfb26cfd2f62c2f05b3b5c7352705d3766cd1e5b27648/pip-1.3.tar.gz#sha256=d6a13c5be316cb21a0243047c7f163f47e88973ebccff8d32e63ca1bf4d9321c (from https://pypi.org/simple/pip/), version: 1.3
      Found link https://files.pythonhosted.org/packages/5b/ce/f5b98104f1c10d868936c25f7c597f492d4371aa9ad5fb61a94954ee7208/pip-1.3.1.tar.gz#sha256=145eaa5d1ea1b062663da1f3a97780d7edea4c63c68a37c463b1deedf7bb4957 (from https://pypi.org/simple/pip/), version: 1.3.1
      Found link https://files.pythonhosted.org/packages/5f/d0/3b3958f6a58783bae44158b2c4c7827ae89abaecdd4bed12cff402620b9a/pip-1.4.tar.gz#sha256=1fd43cbf07d95ddcecbb795c97a1674b3ddb711bb4a67661284a5aa765aa1b97 (from https://pypi.org/simple/pip/), version: 1.4
      Found link https://files.pythonhosted.org/packages/3f/f8/da390e0df72fb61d176b25a4b95262e3dcc14bda0ad25ac64d56db38b667/pip-1.4.1.tar.gz#sha256=4e7a06554711a624c35d0c646f63674b7f6bfc7f80221bf1eb1f631bd890d04e (from https://pypi.org/simple/pip/), version: 1.4.1
      Found link https://files.pythonhosted.org/packages/4f/7d/e53bc80667378125a9e07d4929a61b0bd7128a1129dbe6f07bb3228652a3/pip-1.5.tar.gz#sha256=25f81d1a0e55d3b1709818dd57fdfb954b028f229f09bd69cb0bc80a8e03e048 (from https://pypi.org/simple/pip/), version: 1.5
      Config variable 'Py_DEBUG' is unset, Python ABI tag may be incorrect
      Config variable 'WITH_PYMALLOC' is unset, Python ABI tag may be incorrect
      Found link https://files.pythonhosted.org/packages/44/5d/1dca53b5de6d287e7eb99bd174bb022eb6cb0d6ca6e19ca6b16655dde8c2/pip-1.5.1-py2.py3-none-any.whl#sha256=00960db3b0b8724dd37fe37cfb9c72ecb8f59fab9db7d17c5c1e89a1adab49ce (from https://pypi.org/simple/pip/), version: 1.5.1
      Found link https://files.pythonhosted.org/packages/21/3f/d86a600c9b2f41a75caacf768a24130f343def97652de2345da15ef7911f/pip-1.5.1.tar.gz#sha256=e60e936fbc101d56668c6134c1f2b5b40fcbec8b4fc4ca7fc34842b6b4c5c130 (from https://pypi.org/simple/pip/), version: 1.5.1
      Found link https://files.pythonhosted.org/packages/3d/1f/227d77d5e9ed2df5162de4ba3616799a351eccb1ecd668ae824dd26153a1/pip-1.5.2-py2.py3-none-any.whl#sha256=6903909ccdcdbc3297b74118590e71344d6d262827acd1f5c0e2fcfce9807499 (from https://pypi.org/simple/pip/), version: 1.5.2
      Found link https://files.pythonhosted.org/packages/ed/94/391a003107f6ec997c314199d03bff1c105af758ee490e3255353574487b/pip-1.5.2.tar.gz#sha256=2a8a3e08e652d3a40edbb39264bf01f8ff3c32520a79113357cca1f30533f738 (from https://pypi.org/simple/pip/), version: 1.5.2
      Found link https://files.pythonhosted.org/packages/df/e9/bdb53d44fad1465b43edaf6bc7dd3027ed5af81405cc97603fdff0721ebb/pip-1.5.3-py2.py3-none-any.whl#sha256=f0037aed3ce6cf96b9e9117d42e967a74bea9ebe19088a2fdea5de93d5762fee (from https://pypi.org/simple/pip/), version: 1.5.3
      Found link https://files.pythonhosted.org/packages/55/de/671a48ad313c808623041fc475f7c8f7610401d9f573f06b40eeb84e74e3/pip-1.5.3.tar.gz#sha256=dc53b4d28b88556a37cd73052b6d1d08cc644c6724e37c4d38a2e3c03c5440b2 (from https://pypi.org/simple/pip/), version: 1.5.3
      Found link https://files.pythonhosted.org/packages/a9/9a/9aa19fe00de4c025562e5fb3796ff8520165a7dd1a5662c6ec9816e1ae99/pip-1.5.4-py2.py3-none-any.whl#sha256=fb7282556a42e84464f2e963a859ac4012d8134ba6218b70c1d82d145fcfa82f (from https://pypi.org/simple/pip/), version: 1.5.4
      Found link https://files.pythonhosted.org/packages/78/d8/6e58a7130d457edadb753a0ea5708e411c100c7e94e72ad4802feeef735c/pip-1.5.4.tar.gz#sha256=70208a250bb4afdbbdd74c3ac35d4ab9ba1eb6852d02567a6a87f2f5104e30b9 (from https://pypi.org/simple/pip/), version: 1.5.4
      Found link https://files.pythonhosted.org/packages/ce/c2/10d996b9c51b126a9f0bb9e14a9edcdd5c88888323c0685bb9b392b6c47c/pip-1.5.5-py2.py3-none-any.whl#sha256=fe7a5808190067b2598d85def9b83db46e5d64a00848ad843e107c36e1db4ae6 (from https://pypi.org/simple/pip/), version: 1.5.5
      Found link https://files.pythonhosted.org/packages/88/01/a442fde40bd9aaf837612536f16ab751fac628807fd718690795b8ade77d/pip-1.5.5.tar.gz#sha256=4b7f5124364ae9b5ba833dcd8813a84c1c06fba1d7c8543323c7af4b33188eca (from https://pypi.org/simple/pip/), version: 1.5.5
      Found link https://files.pythonhosted.org/packages/3f/08/7347ca4021e7fe0f1ab8f93cbc7d2a7a7350012300ad0e0227d55625e2b8/pip-1.5.6-py2.py3-none-any.whl#sha256=fbc1351ffedf09ca7560428758845a88d648b9730b63ce9e5df53a7c89f039a4 (from https://pypi.org/simple/pip/), version: 1.5.6
      Found link https://files.pythonhosted.org/packages/45/db/4fb9a456b4ec4d3b701456ef562b9d72d76b6358e0c1463d17db18c5b772/pip-1.5.6.tar.gz#sha256=b1a4ae66baf21b7eb05a5e4f37c50c2706fa28ea1f8780ce8efe14dcd9f1726c (from https://pypi.org/simple/pip/), version: 1.5.6
      Found link https://files.pythonhosted.org/packages/dc/7c/21191b5944b917b66e4e4e06d74f668d814b6e8a3ff7acd874479b6f6b3d/pip-6.0-py2.py3-none-any.whl#sha256=5ec6732505bd8be49fe1f8ad557b88253ffb085736396df4d6bea753fc2a8f2c (from https://pypi.org/simple/pip/), version: 6.0
      Found link https://files.pythonhosted.org/packages/38/fd/065c66a88398f240e344fdf496b9707f92d75f88eedc3d10ff847b28a657/pip-6.0.tar.gz#sha256=6103897f1bb68d3f933edd60f3e3830c4ea6b8abf7a4b500db148921b11f6c9b (from https://pypi.org/simple/pip/), version: 6.0
      Found link https://files.pythonhosted.org/packages/e9/7a/cdbc1a12ed52410d557e48d4646f4543e9e991ff32d2374dc6db849aa617/pip-6.0.1-py2.py3-none-any.whl#sha256=322aea7d1f7b9ee68ad87ac4704cad5df97f77e70668c0bd18f964c5daa78173 (from https://pypi.org/simple/pip/), version: 6.0.1
      Found link https://files.pythonhosted.org/packages/4d/c3/8675b90cd89b9b222062f4f6c7e9d48b0387f5b35cbf747a74403a883e56/pip-6.0.1.tar.gz#sha256=fa2f7c68da4a405d673aa38542f9df009d60026db4f532429ac9cbfbda1f959d (from https://pypi.org/simple/pip/), version: 6.0.1
      Found link https://files.pythonhosted.org/packages/71/3c/b5a521e5e99cfff091e282231591f21193fd80de079ec5fb8ed9c6614044/pip-6.0.2-py2.py3-none-any.whl#sha256=7d17b0f267f7c9cd17cd2924bbbe2b4a3d407322c0e09084ca3f1295c1fed50d (from https://pypi.org/simple/pip/), version: 6.0.2
      Found link https://files.pythonhosted.org/packages/4c/5a/f9e8e3de0153282c7cb54a9b991af225536ac914bac858ca664cf883bb3e/pip-6.0.2.tar.gz#sha256=6fa90667706a679e3dc75b27a51fddafa64401c45e96f8ae6c20978183290077 (from https://pypi.org/simple/pip/), version: 6.0.2
      Found link https://files.pythonhosted.org/packages/73/cb/3eebf42003791df29219a3dfa1874572aa16114b44c9b1b0ac66bf96e8c0/pip-6.0.3-py2.py3-none-any.whl#sha256=b72655b6ac6aef1c86dd07f51e8ace8d7aabd6a1c4ff88db87155276fa32a073 (from https://pypi.org/simple/pip/), version: 6.0.3
      Found link https://files.pythonhosted.org/packages/ce/63/8d99ae60d11ae1a65f5d4fc39a529a598bd3b8e067132210cb0c4d9e9f74/pip-6.0.3.tar.gz#sha256=b091a35f5fa0faffac0b27b97e1e1e93ffe63b463c2ea8dbde0c1fb987933614 (from https://pypi.org/simple/pip/), version: 6.0.3
      Found link https://files.pythonhosted.org/packages/c5/0e/c974206726542bc495fc7443dd97834a6d14c2f0cba183fcfcd01075225a/pip-6.0.4-py2.py3-none-any.whl#sha256=8dfd95de29a7a3bb1e7d368cc83d566938eb210b04d553ebfe5e3a422f4aec65 (from https://pypi.org/simple/pip/), version: 6.0.4
      Found link https://files.pythonhosted.org/packages/02/a1/c90f19910ee153d7a0efca7216758121118d7e93084276541383fe9ca82e/pip-6.0.4.tar.gz#sha256=1dbbff9c369e510c7468ab68ba52c003f68f83c99c2f8259acd51099e8799f1e (from https://pypi.org/simple/pip/), version: 6.0.4
      Found link https://files.pythonhosted.org/packages/e9/1b/c6a375a337fb576784cdea3700f6c3eaf1420f0a01458e6e034cc178a84a/pip-6.0.5-py2.py3-none-any.whl#sha256=b2c20e3a2a43b2bbb1d19ad98be27eccc7b0f0ece016da602ccaa757a862b0e2 (from https://pypi.org/simple/pip/), version: 6.0.5
      Found link https://files.pythonhosted.org/packages/19/f2/58628768f618c8c9fea878e0fb97730c0b8a838d3ab3f325768bf12dac94/pip-6.0.5.tar.gz#sha256=3bf42d28be9085ab2e9aecfd69a6da2d31563fe833304bf71a620a30c38ab8a2 (from https://pypi.org/simple/pip/), version: 6.0.5
      Found link https://files.pythonhosted.org/packages/64/fc/4a49ccb18f55a0ceeb76e8d554bd4563217117492997825d194ed0017cc1/pip-6.0.6-py2.py3-none-any.whl#sha256=fb04f8afe1ba57626783f0c8e2f3d46bbaebaa446fcf124f434e968a2fee595e (from https://pypi.org/simple/pip/), version: 6.0.6
      Found link https://files.pythonhosted.org/packages/f6/ce/d9e4e178b66c766c117f62ddf4fece019ef9d50127a8926d2f60300d615e/pip-6.0.6.tar.gz#sha256=3a14091299dcdb9bab9e9004ae67ac401f2b1b14a7c98de074ca74fdddf4bfa0 (from https://pypi.org/simple/pip/), version: 6.0.6
      Found link https://files.pythonhosted.org/packages/7a/8e/2bbd4fcf3ee06ee90ded5f39ec12f53165dfdb9ef25a981717ad38a16670/pip-6.0.7-py2.py3-none-any.whl#sha256=93a326304c7db749896bcef822bbbac1ab29dad5651c6d732e245975239890e6 (from https://pypi.org/simple/pip/), version: 6.0.7
      Found link https://files.pythonhosted.org/packages/52/85/b160ebdaa84378df6bb0176d4eed9f57edca662446174eead7a9e2e566d6/pip-6.0.7.tar.gz#sha256=35a5a43ac6b7af83ed47ea5731a365f43d350a3a7267e039e5f06b61d42ab3c2 (from https://pypi.org/simple/pip/), version: 6.0.7
      Found link https://files.pythonhosted.org/packages/63/65/55b71647adec1ad595bf0e5d76d028506dfc002df30c256f022ff7a660a5/pip-6.0.8-py2.py3-none-any.whl#sha256=3c22b0a8ff92727bd737a82f72700790591f177541df08c07bc1f90d6b72ac19 (from https://pypi.org/simple/pip/), version: 6.0.8
      Found link https://files.pythonhosted.org/packages/ef/8a/e3a980bc0a7f791d72c1302f65763ed300f2e14c907ac033e01b44c79e5e/pip-6.0.8.tar.gz#sha256=0d58487a1b7f5be2e5e965c11afbea1dc44ecec8069de03491a4d0d6c85f4551 (from https://pypi.org/simple/pip/), version: 6.0.8
      Found link https://files.pythonhosted.org/packages/24/fb/8a56a46243514681e569bbafd8146fa383476c4b7c725c8598c452366f31/pip-6.1.0-py2.py3-none-any.whl#sha256=435a018f6d29e34d4f901bf4e6860d8a5fa1816b68d62008c18ca062a306db31 (from https://pypi.org/simple/pip/), version: 6.1.0
      Found link https://files.pythonhosted.org/packages/6c/84/432eb60bbcb414b9cdfcb135d5f4925e253c74e7d6916ada79990d6cc1a0/pip-6.1.0.tar.gz#sha256=89f120e2ab3d25ab70c36eb28ad4f280fc9ba71736e74d3055f609c1f9173768 (from https://pypi.org/simple/pip/), version: 6.1.0
      Found link https://files.pythonhosted.org/packages/67/f0/ba0fb41dbdbfc4aa3e0c16b40269aca6b9e3d59cacdb646218aa2e9b1d2c/pip-6.1.1-py2.py3-none-any.whl#sha256=a67e54aa0f26b6d62ccec5cc6735eff205dd0fed075f56ac3d3111e91e4467fc (from https://pypi.org/simple/pip/), version: 6.1.1
      Found link https://files.pythonhosted.org/packages/bf/85/871c126b50b8ee0b9819e8a63b614aedd264577e73478caedcd447e8f28c/pip-6.1.1.tar.gz#sha256=89f3b626d225e08e7f20d85044afa40f612eb3284484169813dc2d0631f2a556 (from https://pypi.org/simple/pip/), version: 6.1.1
      Found link https://files.pythonhosted.org/packages/5a/9b/56d3c18d0784d5f2bbd446ea2dc7ffa7476c35e3dc223741d20cfee3b185/pip-7.0.0-py2.py3-none-any.whl#sha256=309c48399c7d68501a10ef206abd6e5c541fedbf84b95435d9063bd454b39df7 (from https://pypi.org/simple/pip/), version: 7.0.0
      Found link https://files.pythonhosted.org/packages/c6/16/6475b142927ca5d03e3b7968efa5b0edd103e4684ecfde181a25f6fa2505/pip-7.0.0.tar.gz#sha256=7b46bfc1b95494731de306a688e2a7bc056d7fa7ad27e026908fb2ae67fed23d (from https://pypi.org/simple/pip/), version: 7.0.0
      Found link https://files.pythonhosted.org/packages/5a/10/bb7a32c335bceba636aa673a4c977effa1e73a79f88856459486d8d670cf/pip-7.0.1-py2.py3-none-any.whl#sha256=d26b8573ba1ac1ec99a9bdbdffee2ff2b06c7790815211d0eb4dc1462a089705 (from https://pypi.org/simple/pip/), version: 7.0.1
      Found link https://files.pythonhosted.org/packages/4a/83/9ae4362a80739657e0c8bb628ea3fa0214a9aba7c8590dacc301ea293f73/pip-7.0.1.tar.gz#sha256=cfec177552fdd0b2d12b72651c8e874f955b4c62c1c2c9f2588cbdc1c0d0d416 (from https://pypi.org/simple/pip/), version: 7.0.1
      Found link https://files.pythonhosted.org/packages/64/7f/7107800ae0919a80afbf1ecba21b90890431c3ee79d700adac3c79cb6497/pip-7.0.2-py2.py3-none-any.whl#sha256=83c869c5ab7113866e2d69641ec470d47f0faae68ca4550a289a4d3db515ad65 (from https://pypi.org/simple/pip/), version: 7.0.2
      Found link https://files.pythonhosted.org/packages/75/b1/66532c273bca0133e42c3b4540a1609289f16e3046f1830f18c60794d661/pip-7.0.2.tar.gz#sha256=ba28fa60b573a9444e7b78ccb3b0f261d1f66f46d20403f9dce37b18a6aed405 (from https://pypi.org/simple/pip/), version: 7.0.2
      Found link https://files.pythonhosted.org/packages/96/76/33a598ae42dd0554207d83c7acc60e3b166dbde723cbf282f1f73b7a127c/pip-7.0.3-py2.py3-none-any.whl#sha256=7b1cb03e827d58d2d05e68ea96a9e27487ed4b0afcd951ac6e40847ce94f0738 (from https://pypi.org/simple/pip/), version: 7.0.3
      Found link https://files.pythonhosted.org/packages/35/59/5b23115758ba0f2fc465c459611865173ef006202ba83f662d1f58ed2fb8/pip-7.0.3.tar.gz#sha256=b4c598825a6f6dc2cac65968feb28e6be6c1f7f1408493c60a07eaa731a0affd (from https://pypi.org/simple/pip/), version: 7.0.3
      Found link https://files.pythonhosted.org/packages/f7/c0/9f8dac88326609b4b12b304e8382f64f7d5af7735a00d2fac36cf135fc30/pip-7.1.0-py2.py3-none-any.whl#sha256=80c29f899d3a00a448d65f8158544d22935baec7159af8da1a4fa1490ced481d (from https://pypi.org/simple/pip/), version: 7.1.0
      Found link https://files.pythonhosted.org/packages/7e/71/3c6ece07a9a885650aa6607b0ebfdf6fc9a3ef8691c44b5e724e4eee7bf2/pip-7.1.0.tar.gz#sha256=d5275ba3221182a5dd1b6bcfbfc5ec277fb399dd23226d6fa018048f7e0f10f2 (from https://pypi.org/simple/pip/), version: 7.1.0
      Found link https://files.pythonhosted.org/packages/1c/56/094d563c508917081bccff365e4f621ba33073c1c13aca9267a43cfcaf13/pip-7.1.1-py2.py3-none-any.whl#sha256=ce13000878d34c1178af76cb8cf269e232c00508c78ed46c165dd5b0881615f4 (from https://pypi.org/simple/pip/), version: 7.1.1
      Found link https://files.pythonhosted.org/packages/3b/bb/b3f2a95494fd3f01d3b3ae530e7c0e910dc25e88e30787b0a5e10cbc0640/pip-7.1.1.tar.gz#sha256=b22fe3c93a13fc7c04f145a42fd2ad50a9e3e1b8a7eed2e2b1c66e540a0951da (from https://pypi.org/simple/pip/), version: 7.1.1
      Found link https://files.pythonhosted.org/packages/b2/d0/cd115fe345dd6f07ec1c780020a7dfe74966fceeb171e0f20d1d4905b0b7/pip-7.1.2-py2.py3-none-any.whl#sha256=b9d3983b5cce04f842175e30169d2f869ef12c3546fd274083a65eada4e9708c (from https://pypi.org/simple/pip/), version: 7.1.2
      Found link https://files.pythonhosted.org/packages/d0/92/1e8406c15d9372084a5bf79d96da3a0acc4e7fcf0b80020a4820897d2a5c/pip-7.1.2.tar.gz#sha256=ca047986f0528cfa975a14fb9f7f106271d4e0c3fe1ddced6c1db2e7ae57a477 (from https://pypi.org/simple/pip/), version: 7.1.2
      Found link https://files.pythonhosted.org/packages/00/ae/bddef02881ee09c6a01a0d6541aa6c75a226a4e68b041be93142befa0cd6/pip-8.0.0-py2.py3-none-any.whl#sha256=262ed1823eb7fbe3f18a9bedb4800e59c4ab9a6682aff8c37b5ee83ea840910b (from https://pypi.org/simple/pip/), version: 8.0.0
      Found link https://files.pythonhosted.org/packages/e3/2d/03c014d11e66628abf2fda5ca00f779cbe7b5292c5cd13d42a95b94aa9b8/pip-8.0.0.tar.gz#sha256=90112b296152f270cb8dddcd19b7b87488d9e002e8cf622e14c4da9c2f6319b1 (from https://pypi.org/simple/pip/), version: 8.0.0
      Found link https://files.pythonhosted.org/packages/45/9c/6f9a24917c860873e2ce7bd95b8f79897524353df51d5d920cd6b6c1ec33/pip-8.0.1-py2.py3-none-any.whl#sha256=dedaac846bc74e38a3253671f51a056331ffca1da70e3f48d8128f2aa0635bba (from https://pypi.org/simple/pip/), version: 8.0.1
      Found link https://files.pythonhosted.org/packages/ea/66/a3d6187bd307159fedf8575c0d9ee2294d13b1cdd11673ca812e6a2dda8f/pip-8.0.1.tar.gz#sha256=477c50b3e538a7ac0fa611fb8b877b04b33fb70d325b12a81b9dbf3eb1158a4d (from https://pypi.org/simple/pip/), version: 8.0.1
      Found link https://files.pythonhosted.org/packages/e7/a0/bd35f5f978a5e925953ce02fa0f078a232f0f10fcbe543d8cfc043f74fda/pip-8.0.2-py2.py3-none-any.whl#sha256=249a6f3194be8c2e8cb4d4be3f6fd16a9f1e3336218caffa8e7419e3816f9988 (from https://pypi.org/simple/pip/), version: 8.0.2
      Found link https://files.pythonhosted.org/packages/ce/15/ee1f9a84365423e9ef03d0f9ed0eba2fb00ac1fffdd33e7b52aea914d0f8/pip-8.0.2.tar.gz#sha256=46f4bd0d8dfd51125a554568d646fe4200a3c2c6c36b9f2d06d2212148439521 (from https://pypi.org/simple/pip/), version: 8.0.2
      Found link https://files.pythonhosted.org/packages/ae/d4/2b127310f5364610b74c28e2e6a40bc19e2d3c9a9a4e012d3e333e767c99/pip-8.0.3-py2.py3-none-any.whl#sha256=b0335bc837f9edb5aad03bd43d0973b084a1cbe616f8188dc23ba13234dbd552 (from https://pypi.org/simple/pip/), version: 8.0.3
      Found link https://files.pythonhosted.org/packages/22/f3/14bc87a4f6b5ec70b682765978a6f3105bf05b6781fa97e04d30138bd264/pip-8.0.3.tar.gz#sha256=30f98b66f3fe1069c529a491597d34a1c224a68640c82caf2ade5f88aa1405e8 (from https://pypi.org/simple/pip/), version: 8.0.3
      Found link https://files.pythonhosted.org/packages/1e/c7/78440b3fb882ed001e6e12d8770bd45e73d6eced4e57f7c072b829ce8a3d/pip-8.1.0-py2.py3-none-any.whl#sha256=a542b99e08002ead83200198e19a3983270357e1cb4fe704247990b5b35471dc (from https://pypi.org/simple/pip/), version: 8.1.0
      Found link https://files.pythonhosted.org/packages/3c/72/6981d5adf880adecb066a1a1a4c312a17f8d787a3b85446967964ac66d55/pip-8.1.0.tar.gz#sha256=d8faa75dd7d0737b16d50cd0a56dc91a631c79ecfd8d38b80f6ee929ec82043e (from https://pypi.org/simple/pip/), version: 8.1.0
      Found link https://files.pythonhosted.org/packages/31/6a/0f19a7edef6c8e5065f4346137cc2a08e22e141942d66af2e1e72d851462/pip-8.1.1-py2.py3-none-any.whl#sha256=44b9c342782ab905c042c207d995aa069edc02621ddbdc2b9f25954a0fdac25c (from https://pypi.org/simple/pip/), version: 8.1.1
      Found link https://files.pythonhosted.org/packages/41/27/9a8d24e1b55bd8c85e4d022da2922cb206f183e2d18fee4e320c9547e751/pip-8.1.1.tar.gz#sha256=3e78d3066aaeb633d185a57afdccf700aa2e660436b4af618bcb6ff0fa511798 (from https://pypi.org/simple/pip/), version: 8.1.1
      Found link https://files.pythonhosted.org/packages/9c/32/004ce0852e0a127f07f358b715015763273799bd798956fa930814b60f39/pip-8.1.2-py2.py3-none-any.whl#sha256=6464dd9809fb34fc8df2bf49553bb11dac4c13d2ffa7a4f8038ad86a4ccb92a1 (from https://pypi.org/simple/pip/), version: 8.1.2
      Found link https://files.pythonhosted.org/packages/e7/a8/7556133689add8d1a54c0b14aeff0acb03c64707ce100ecd53934da1aa13/pip-8.1.2.tar.gz#sha256=4d24b03ffa67638a3fa931c09fd9e0273ffa904e95ebebe7d4b1a54c93d7b732 (from https://pypi.org/simple/pip/), version: 8.1.2
      Found link https://files.pythonhosted.org/packages/3f/ef/935d9296acc4f48d1791ee56a73781271dce9712b059b475d3f5fa78487b/pip-9.0.0-py2.py3-none-any.whl#sha256=c856ac18ca01e7127456f831926dc67cc7d3ab663f4c13b1ec156e36db4de574 (from https://pypi.org/simple/pip/) (requires-python:>=2.6,!=3.0.*,!=3.1.*,!=3.2.*), version: 9.0.0
      Found link https://files.pythonhosted.org/packages/5e/53/eaef47e5e2f75677c9de0737acc84b659b78a71c4086f424f55346a341b5/pip-9.0.0.tar.gz#sha256=f62fb70e7e000e46fce12aaeca752e5281a5446977fe5a75ab4189a43b3f8793 (from https://pypi.org/simple/pip/) (requires-python:>=2.6,!=3.0.*,!=3.1.*,!=3.2.*), version: 9.0.0
      Found link https://files.pythonhosted.org/packages/b6/ac/7015eb97dc749283ffdec1c3a88ddb8ae03b8fad0f0e611408f196358da3/pip-9.0.1-py2.py3-none-any.whl#sha256=690b762c0a8460c303c089d5d0be034fb15a5ea2b75bdf565f40421f542fefb0 (from https://pypi.org/simple/pip/) (requires-python:>=2.6,!=3.0.*,!=3.1.*,!=3.2.*), version: 9.0.1
      Found link https://files.pythonhosted.org/packages/11/b6/abcb525026a4be042b486df43905d6893fb04f05aac21c32c638e939e447/pip-9.0.1.tar.gz#sha256=09f243e1a7b461f654c26a725fa373211bb7ff17a9300058b205c61658ca940d (from https://pypi.org/simple/pip/) (requires-python:>=2.6,!=3.0.*,!=3.1.*,!=3.2.*), version: 9.0.1
      Found link https://files.pythonhosted.org/packages/e7/f9/e801dcea22886cd513f6bd2e8f7e581bd6f67bb8e8f1cd8e7b92d8539280/pip-9.0.2-py2.py3-none-any.whl#sha256=b135491ddb061f39719b8472d8abb59c613816a2b86069c332db74d1cd208ab2 (from https://pypi.org/simple/pip/) (requires-python:>=2.6,!=3.0.*,!=3.1.*,!=3.2.*), version: 9.0.2
      Found link https://files.pythonhosted.org/packages/e5/8f/3fc66461992dc9e9fcf5e005687d5f676729172dda640df2fd8b597a6da7/pip-9.0.2.tar.gz#sha256=88110a224e9d30e5d76592a0b2130ef10e7e67a6426e8617bb918fffbfe91fe5 (from https://pypi.org/simple/pip/) (requires-python:>=2.6,!=3.0.*,!=3.1.*,!=3.2.*), version: 9.0.2
      Found link https://files.pythonhosted.org/packages/ac/95/a05b56bb975efa78d3557efa36acaf9cf5d2fd0ee0062060493687432e03/pip-9.0.3-py2.py3-none-any.whl#sha256=c3ede34530e0e0b2381e7363aded78e0c33291654937e7373032fda04e8803e5 (from https://pypi.org/simple/pip/) (requires-python:>=2.6,!=3.0.*,!=3.1.*,!=3.2.*), version: 9.0.3
      Found link https://files.pythonhosted.org/packages/c4/44/e6b8056b6c8f2bfd1445cc9990f478930d8e3459e9dbf5b8e2d2922d64d3/pip-9.0.3.tar.gz#sha256=7bf48f9a693be1d58f49f7af7e0ae9fe29fd671cde8a55e6edca3581c4ef5796 (from https://pypi.org/simple/pip/) (requires-python:>=2.6,!=3.0.*,!=3.1.*,!=3.2.*), version: 9.0.3
      Found link https://files.pythonhosted.org/packages/4b/5a/8544ae02a5bd28464e03af045e8aabde20a7b02db1911a9159328e1eb25a/pip-10.0.0b1-py2.py3-none-any.whl#sha256=dbd5d24cd461be23429625085a36cc8732cbcac4d2aaf673031f80f6ac07d844 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*), version: 10.0.0b1
      Found link https://files.pythonhosted.org/packages/aa/6d/ffbb86abf18b750fb26f27eda7c7732df2aacaa669c420d2eb2ad6df3458/pip-10.0.0b1.tar.gz#sha256=8d6e63d8b99752e4b53f272b66f9cd7b59e2b288e9a863a61c48d167203a2656 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*), version: 10.0.0b1
      Found link https://files.pythonhosted.org/packages/97/72/1d514201e7d7fc7fff5aac3de9c7b892cd72fb4bf23fd983630df96f7412/pip-10.0.0b2-py2.py3-none-any.whl#sha256=79f55588912f1b2b4f86f96f11e329bb01b25a484e2204f245128b927b1038a7 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*), version: 10.0.0b2
      Found link https://files.pythonhosted.org/packages/32/67/572f642e6e42c580d3154964cfbab7d9322c23b0f417c6c01fdd206a2777/pip-10.0.0b2.tar.gz#sha256=ad6adec2150ce4aed8f6134d9b77d928fc848dbcb887fb1a455988cf99da5cae (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*), version: 10.0.0b2
      Found link https://files.pythonhosted.org/packages/62/a1/0d452b6901b0157a0134fd27ba89bf95a857fbda64ba52e1ca2cf61d8412/pip-10.0.0-py2.py3-none-any.whl#sha256=86a60a96d85e329962a9e6f6af612cbc11106293dbc83f119802b5bee9874cf3 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*), version: 10.0.0
      Found link https://files.pythonhosted.org/packages/e0/69/983a8e47d3dfb51e1463c1e962b2ccd1d74ec4e236e232625e353d830ed2/pip-10.0.0.tar.gz#sha256=f05a3eeea64bce94e85cc6671d679473d66288a4d37c3fcf983584954096b34f (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*), version: 10.0.0
      Found link https://files.pythonhosted.org/packages/0f/74/ecd13431bcc456ed390b44c8a6e917c1820365cbebcb6a8974d1cd045ab4/pip-10.0.1-py2.py3-none-any.whl#sha256=717cdffb2833be8409433a93746744b59505f42146e8d37de6c62b430e25d6d7 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*), version: 10.0.1
      Found link https://files.pythonhosted.org/packages/ae/e8/2340d46ecadb1692a1e455f13f75e596d4eab3d11a57446f08259dee8f02/pip-10.0.1.tar.gz#sha256=f2bd08e0cd1b06e10218feaf6fef299f473ba706582eb3bd9d52203fdbd7ee68 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*), version: 10.0.1
      Found link https://files.pythonhosted.org/packages/5f/25/e52d3f31441505a5f3af41213346e5b6c221c9e086a166f3703d2ddaf940/pip-18.0-py2.py3-none-any.whl#sha256=070e4bf493c7c2c9f6a08dd797dd3c066d64074c38e9e8a0fb4e6541f266d96c (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 18.0
      Found link https://files.pythonhosted.org/packages/69/81/52b68d0a4de760a2f1979b0931ba7889202f302072cc7a0d614211bc7579/pip-18.0.tar.gz#sha256=a0e11645ee37c90b40c46d607070c4fd583e2cd46231b1c06e389c5e814eed76 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 18.0
      Found link https://files.pythonhosted.org/packages/c2/d7/90f34cb0d83a6c5631cf71dfe64cc1054598c843a92b400e55675cc2ac37/pip-18.1-py2.py3-none-any.whl#sha256=7909d0a0932e88ea53a7014dfd14522ffef91a464daaaf5c573343852ef98550 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 18.1
      Found link https://files.pythonhosted.org/packages/45/ae/8a0ad77defb7cc903f09e551d88b443304a9bd6e6f124e75c0fbbf6de8f7/pip-18.1.tar.gz#sha256=c0a292bd977ef590379a3f05d7b7f65135487b67470f6281289a94e015650ea1 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 18.1
      Found link https://files.pythonhosted.org/packages/60/64/73b729587b6b0d13e690a7c3acd2231ee561e8dd28a58ae1b0409a5a2b20/pip-19.0-py2.py3-none-any.whl#sha256=249ab0de4c1cef3dba4cf3f8cca722a07fc447b1692acd9f84e19c646db04c9a (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.0
      Found link https://files.pythonhosted.org/packages/11/31/c483614095176ddfa06ac99c2af4171375053b270842c7865ca0b4438dc1/pip-19.0.tar.gz#sha256=c82bf8bc00c5732f0dd49ac1dea79b6242a1bd42a5012e308ed4f04369b17e54 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.0
      Found link https://files.pythonhosted.org/packages/46/dc/7fd5df840efb3e56c8b4f768793a237ec4ee59891959d6a215d63f727023/pip-19.0.1-py2.py3-none-any.whl#sha256=aae79c7afe895fb986ec751564f24d97df1331bb99cdfec6f70dada2f40c0044 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.0.1
      Found link https://files.pythonhosted.org/packages/c8/89/ad7f27938e59db1f0f55ce214087460f65048626e2226531ba6cb6da15f0/pip-19.0.1.tar.gz#sha256=e81ddd35e361b630e94abeda4a1eddd36d47a90e71eb00f38f46b57f787cd1a5 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.0.1
      Found link https://files.pythonhosted.org/packages/d7/41/34dd96bd33958e52cb4da2f1bf0818e396514fd4f4725a79199564cd0c20/pip-19.0.2-py2.py3-none-any.whl#sha256=6a59f1083a63851aeef60c7d68b119b46af11d9d803ddc1cf927b58edcd0b312 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.0.2
      Found link https://files.pythonhosted.org/packages/4c/4d/88bc9413da11702cbbace3ccc51350ae099bb351febae8acc85fec34f9af/pip-19.0.2.tar.gz#sha256=f851133f8b58283fa50d8c78675eb88d4ff4cde29b6c41205cd938b06338e0e5 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.0.2
      Found link https://files.pythonhosted.org/packages/d8/f3/413bab4ff08e1fc4828dfc59996d721917df8e8583ea85385d51125dceff/pip-19.0.3-py2.py3-none-any.whl#sha256=bd812612bbd8ba84159d9ddc0266b7fbce712fc9bc98c82dee5750546ec8ec64 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.0.3
      Found link https://files.pythonhosted.org/packages/36/fa/51ca4d57392e2f69397cd6e5af23da2a8d37884a605f9e3f2d3bfdc48397/pip-19.0.3.tar.gz#sha256=6e6f197a1abfb45118dbb878b5c859a0edbdd33fd250100bc015b67fded4b9f2 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.0.3
      Found link https://files.pythonhosted.org/packages/f9/fb/863012b13912709c13cf5cfdbfb304fa6c727659d6290438e1a88df9d848/pip-19.1-py2.py3-none-any.whl#sha256=8f59b6cf84584d7962d79fd1be7a8ec0eb198aa52ea864896551736b3614eee9 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.1
      Found link https://files.pythonhosted.org/packages/51/5f/802a04274843f634469ef299fcd273de4438386deb7b8681dd059f0ee3b7/pip-19.1.tar.gz#sha256=d9137cb543d8a4d73140a3282f6d777b2e786bb6abb8add3ac5b6539c82cd624 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.1
      Found link https://files.pythonhosted.org/packages/5c/e0/be401c003291b56efc55aeba6a80ab790d3d4cece2778288d65323009420/pip-19.1.1-py2.py3-none-any.whl#sha256=993134f0475471b91452ca029d4390dc8f298ac63a712814f101cd1b6db46676 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.1.1
      Found link https://files.pythonhosted.org/packages/93/ab/f86b61bef7ab14909bd7ec3cd2178feb0a1c86d451bc9bccd5a1aedcde5f/pip-19.1.1.tar.gz#sha256=44d3d7d3d30a1eb65c7e5ff1173cdf8f7467850605ac7cc3707b6064bddd0958 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*), version: 19.1.1
      Found link https://files.pythonhosted.org/packages/3a/6f/35de4f49ae5c7fdb2b64097ab195020fb48faa8ad3a85386ece6953c11b1/pip-19.2-py2.py3-none-any.whl#sha256=468c67b0b1120cd0329dc72972cf0651310783a922e7609f3102bd5fb4acbf17 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.2
      Found link https://files.pythonhosted.org/packages/41/13/b6e68eae78405af6e4e9a93319ae5bb371057786f1590b157341f7542d7d/pip-19.2.tar.gz#sha256=aa6fdd80d13caac75d92b5eced06778712859b1606ba92d62389c11be12b2dad (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.2
      Found link https://files.pythonhosted.org/packages/62/ca/94d32a6516ed197a491d17d46595ce58a83cbb2fca280414e57cd86b84dc/pip-19.2.1-py2.py3-none-any.whl#sha256=80d7452630a67c1e7763b5f0a515690f2c1e9ad06dda48e0ae85b7fdf2f59d97 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.2.1
      Found link https://files.pythonhosted.org/packages/8b/8a/1b2aadd922db1afe6bc107b03de41d6d37a28a5923383e60695fba24ae81/pip-19.2.1.tar.gz#sha256=258d702483dd749400aec59c23d638a5b2249ae28a0f478b6cab12ad45681a80 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.2.1
      Found link https://files.pythonhosted.org/packages/8d/07/f7d7ced2f97ca3098c16565efbe6b15fafcba53e8d9bdb431e09140514b0/pip-19.2.2-py2.py3-none-any.whl#sha256=4b956bd8b7b481fc5fa222637ff6d0823a327e5118178f1ec47618a480e61997 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.2.2
      Found link https://files.pythonhosted.org/packages/aa/1a/62fb0b95b1572c76dbc3cc31124a8b6866cbe9139eb7659ac7349457cf7c/pip-19.2.2.tar.gz#sha256=e05103825871e210d50a44c7e448587b0ed99dd775d3ef586304c58f40224a53 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.2.2
      Found link https://files.pythonhosted.org/packages/30/db/9e38760b32e3e7f40cce46dd5fb107b8c73840df38f0046d8e6514e675a1/pip-19.2.3-py2.py3-none-any.whl#sha256=340a0ba40fdeb16413914c0fcd8e0b4ebb0bf39a900ec80e11c05d836c05103f (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.2.3
      Found link https://files.pythonhosted.org/packages/00/9e/4c83a0950d8bdec0b4ca72afd2f9cea92d08eb7c1a768363f2ea458d08b4/pip-19.2.3.tar.gz#sha256=e7a31f147974362e6c82d84b91c7f2bdf57e4d3163d3d454e6c3e71944d67135 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.2.3
      Found link https://files.pythonhosted.org/packages/4a/08/6ca123073af4ebc4c5488a5bc8a010ac57aa39ce4d3c8a931ad504de4185/pip-19.3-py2.py3-none-any.whl#sha256=e100a7eccf085f0720b4478d3bb838e1c179b1e128ec01c0403f84e86e0e2dfb (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.3
      Found link https://files.pythonhosted.org/packages/af/7a/5dd1e6efc894613c432ce86f1011fcc3bbd8ac07dfeae6393b7b97f1de8b/pip-19.3.tar.gz#sha256=324d234b8f6124846b4e390df255cacbe09ce22791c3b714aa1ea6e44a4f2861 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.3
      Found link https://files.pythonhosted.org/packages/00/b6/9cfa56b4081ad13874b0c6f96af8ce16cfbc1cb06bedf8e9164ce5551ec1/pip-19.3.1-py2.py3-none-any.whl#sha256=6917c65fc3769ecdc61405d3dfd97afdedd75808d200b2838d7d961cebc0c2c7 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.3.1
      Found link https://files.pythonhosted.org/packages/ce/ea/9b445176a65ae4ba22dce1d93e4b5fe182f953df71a145f557cffaffc1bf/pip-19.3.1.tar.gz#sha256=21207d76c1031e517668898a6b46a9fb1501c7a4710ef5dfd6a40ad9e6757ea7 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 19.3.1
      Found link https://files.pythonhosted.org/packages/60/65/16487a7c4e0f95bb3fc89c2e377be331fd496b7a9b08fd3077de7f3ae2cf/pip-20.0-py2.py3-none-any.whl#sha256=eea07b449d969dbc8c062c157852cf8ed2ad1b8b5ac965a6b819e62929e41703 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 20.0
      Found link https://files.pythonhosted.org/packages/8c/5c/c18d58ab5c1a702bf670e0bd6a77cd4645e4aeca021c6118ef850895cc96/pip-20.0.tar.gz#sha256=5128e9a9401f1d16c1d15b2ed766a79d7813db1538428d0b0ce74838249e3a41 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 20.0
      Found link https://files.pythonhosted.org/packages/57/36/67f809c135c17ec9b8276466cc57f35b98c240f55c780689ea29fa32f512/pip-20.0.1-py2.py3-none-any.whl#sha256=b7110a319790ae17e8105ecd6fe07dbcc098a280c6d27b6dd7a20174927c24d7 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 20.0.1
      Found link https://files.pythonhosted.org/packages/28/af/2c76c8aa46ccdf7578b83d97a11a2d1858794d4be4a1610ade0d30182e8b/pip-20.0.1.tar.gz#sha256=3cebbac2a1502e09265f94e5717408339de846b3c0f0ed086d7b817df9cab822 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 20.0.1
      Found link https://files.pythonhosted.org/packages/54/0c/d01aa759fdc501a58f431eb594a17495f15b88da142ce14b5845662c13f3/pip-20.0.2-py2.py3-none-any.whl#sha256=4ae14a42d8adba3205ebeb38aa68cfc0b6c346e1ae2e699a0b3bad4da19cef5c (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 20.0.2
      Found link https://files.pythonhosted.org/packages/8e/76/66066b7bc71817238924c7e4b448abdb17eb0c92d645769c223f9ace478f/pip-20.0.2.tar.gz#sha256=7db0c8ea4c7ea51c8049640e8e6e7fde949de672bfa4949920675563a5a6967f (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 20.0.2
      Found link https://files.pythonhosted.org/packages/ec/05/82d3fababbf462d876883ebc36f030f4fa057a563a80f5a26ee63679d9ea/pip-20.1b1-py2.py3-none-any.whl#sha256=4cf0348b683937da883ccaae8c8bcfc9b4c7ba4c48b38cc2d89cd7b8d0b220d9 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 20.1b1
      Found link https://files.pythonhosted.org/packages/cd/81/c1184456fe506bd50992571c9f8581907976ce71502e36741f033e2da1f1/pip-20.1b1.tar.gz#sha256=699880a47f6d306f4f9a87ca151ef33d41d2223b81ff343b786d38c297923a19 (from https://pypi.org/simple/pip/) (requires-python:>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*), version: 20.1b1
    Given no hashes to check 137 links for project 'pip': discarding no candidates
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install bpemb
```

    Requirement already satisfied: bpemb in c:\installation\lib\site-packages (0.3.0)
    Requirement already satisfied: sentencepiece in c:\installation\lib\site-packages (from bpemb) (0.1.85)
    Requirement already satisfied: gensim in c:\installation\lib\site-packages (from bpemb) (3.8.1)
    Requirement already satisfied: numpy in c:\installation\lib\site-packages (from bpemb) (1.16.5)
    Requirement already satisfied: tqdm in c:\installation\lib\site-packages (from bpemb) (4.36.1)
    Requirement already satisfied: requests in c:\installation\lib\site-packages (from bpemb) (2.22.0)
    Requirement already satisfied: scipy>=0.18.1 in c:\installation\lib\site-packages (from gensim->bpemb) (1.4.1)
    Requirement already satisfied: smart-open>=1.8.1 in c:\installation\lib\site-packages (from gensim->bpemb) (1.9.0)
    Requirement already satisfied: six>=1.5.0 in c:\installation\lib\site-packages (from gensim->bpemb) (1.12.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\installation\lib\site-packages (from requests->bpemb) (2019.9.11)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in c:\installation\lib\site-packages (from requests->bpemb) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in c:\installation\lib\site-packages (from requests->bpemb) (2.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\installation\lib\site-packages (from requests->bpemb) (1.24.2)
    Requirement already satisfied: boto>=2.32 in c:\installation\lib\site-packages (from smart-open>=1.8.1->gensim->bpemb) (2.49.0)
    Requirement already satisfied: boto3 in c:\installation\lib\site-packages (from smart-open>=1.8.1->gensim->bpemb) (1.12.21)
    Requirement already satisfied: botocore<1.16.0,>=1.15.21 in c:\installation\lib\site-packages (from boto3->smart-open>=1.8.1->gensim->bpemb) (1.15.21)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in c:\installation\lib\site-packages (from boto3->smart-open>=1.8.1->gensim->bpemb) (0.9.5)
    Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in c:\installation\lib\site-packages (from boto3->smart-open>=1.8.1->gensim->bpemb) (0.3.3)
    Requirement already satisfied: docutils<0.16,>=0.10 in c:\installation\lib\site-packages (from botocore<1.16.0,>=1.15.21->boto3->smart-open>=1.8.1->gensim->bpemb) (0.15.2)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in c:\installation\lib\site-packages (from botocore<1.16.0,>=1.15.21->boto3->smart-open>=1.8.1->gensim->bpemb) (2.8.0)
    Note: you may need to restart the kernel to use updated packages.
    
