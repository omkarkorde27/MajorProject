#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


import subprocess

try:
    subprocess.check_output('nvidia-smi')
    print('NVIDIA GPU detected.')
except Exception:
    print('No NVIDIA GPU detected.')


# In[3]:


from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup


# In[4]:


import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim


# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams


# In[6]:


import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from collections import defaultdict
from textwrap import wrap

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[7]:


sns.set(style='whitegrid', palette='muted', font_scale=1.2)
sns.set_palette(sns.color_palette("Paired"))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[8]:


pre_trained_model_ckpt = "bert-base-uncased"


# In[9]:


csv_file_path = 'student_grades_updated.csv'


# In[10]:


data = pd.read_csv(csv_file_path)
data.head()


# In[11]:


sorted(data.Grades.unique())


# In[12]:


data1 = data[data['main_category'].isin(['Education & Reference', 'Science & Mathematics'])]

# If you want to reset the index of the filtered DataFrame
data1.reset_index(drop=True, inplace=True)

data1.to_csv('Education_2000.csv')


# In[13]:


df = pd.read_csv('Education_2000.csv')
df.head()


# In[14]:


def map_grade(grade_assigned):
    grade_assigned = str(grade_assigned)
    if grade_assigned == 'F':
        return 0
    elif grade_assigned == 'C':
        return 1
    elif grade_assigned == 'B':
        return 2
    elif grade_assigned == 'A':
        return 3
    elif grade_assigned == 'O':
        return 4
    
df['Grades'] = df.Grades.apply(map_grade)


# In[15]:


df.head()


# In[16]:


sorted(df.Grades.unique())


# In[17]:


grade_counts = df['Grades'].value_counts()

sns.barplot(x=grade_counts.index, y=grade_counts.values)
plt.xlabel('Grades')
plt.ylabel('Count')
plt.title('Distribution of Grades')
plt.show()


# In[18]:


random_answer_list = df.random_answer_list.to_list(),
target = df.Grades.to_list(),


# In[19]:


tokenizer = BertTokenizer.from_pretrained(pre_trained_model_ckpt)


# In[20]:


# Tokenization and encoding on a Sample text

sample_text = 'Text Processing with Machine Learning is moving at a rapid speed'

tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"Sentence: {sample_text}")
print(f"Tokens: {tokens}")
print(f"TOken IDs : {token_ids}")

encoding = tokenizer.encode_plus(
    sample_text,
    max_length=32,
    truncation=True,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding=True,
    return_attention_mask=True,
    return_tensors='pt')

print(f'Encoding keys: {encoding.keys()}')
print(len(encoding['input_ids'][0]))
print(encoding['input_ids'][0])
print(len(encoding['attention_mask'][0]))
print(encoding['attention_mask'])
print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))


# In[21]:


# EDA of token counts in the dataset

token_length_list = []

for txt in df.random_answer_list:
    #if pd.isna(txt) or txt == "": 
         #continue  # Skip to the next text
    tokens = tokenizer.encode(txt, truncation=True, max_length=512)
    token_length_list.append(len(tokens))
    
sns.distplot(token_length_list)

plt.xlim([0,256])

plt.xlabel('Token count')


# In[22]:


#dataset utility class

df_train, df_test = train_test_split(df, test_size=0.3, random_state = RANDOM_SEED)

df_val, df_test = train_test_split(df_test, test_size = 0.5, random_state= RANDOM_SEED)

print(df_train.shape, df_val.shape, df_test.shape)


# In[23]:


import tensorflow as tf

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[24]:


from transformers import DataCollatorWithPadding
from torch.nn.utils.rnn import pad_sequence

MAX_LEN = 160
BATCH_SIZE = 16

class GradeDataset(Dataset):
    def __init__(self, studentanswers, targets, tokenizer, max_len, include_raw_studentanswer = True):
        self.studentanswers = studentanswers
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_raw_studentanswer = include_raw_studentanswer 

    def __len__(self):
        return len(self.studentanswers)

    def __getitem__(self, item):
        studentanswer = str(self.studentanswers[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            studentanswer,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding='max_length', 
            return_tensors='pt'
        )

        output = {
            'student_input_ids': encoding['input_ids'].flatten(),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long),
            'student_answer': studentanswer
        }
        
        if self.include_raw_studentanswer:
            output['student_answer'] = encoding['input_ids'].flatten()
           
        return output

class CustomCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, batch):
        numerical_features = ['input_ids', 'attention_mask', 'targets']  
        text_features = ['student_answer']  
        
        student_input_ids = torch.stack([item['student_input_ids'] for item in batch])

        batch_encodings = {feature: [item[feature] for item in batch] for feature in numerical_features}
        batch_text = {feature: [item[feature] for item in batch] for feature in text_features}

        # Apply padding and numericalization via the base DataCollator
        padded_batch = super().__call__(batch_encodings)  

        for feature in text_features: 
            student_input_ids = torch.stack([item['student_input_ids'] for item in batch])
        padded_batch['student_input_ids'] = pad_sequence([item['student_input_ids'] for item in batch], batch_first=True, padding_value=0)
        return padded_batch
    
pre_trained_model_ckpt = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pre_trained_model_ckpt)
    
collator = CustomCollatorWithPadding(tokenizer=tokenizer, padding = 'longest')
    
def create_data_loader(df, tokenizer, max_len = MAX_LEN, batch_size = BATCH_SIZE, include_raw_studentanswer = True):
    ds = GradeDataset(
        studentanswers = df.random_answer_list.to_list(),
        targets = df.Grades.to_list(),
        tokenizer = tokenizer,
        max_len = max_len,
        include_raw_studentanswer = include_raw_studentanswer
    )
    return DataLoader(ds, batch_size = batch_size, collate_fn = collator)
    
        


# In[25]:


train_data_loader = create_data_loader(df_train, tokenizer)

val_data_loader = create_data_loader(df_val, tokenizer, include_raw_studentanswer = True)

test_data_loader = create_data_loader(df_test, tokenizer, include_raw_studentanswer = True)


# In[26]:


for d in val_data_loader:
    print(d.keys())


# In[27]:


data = next(iter(train_data_loader))

print(data.keys())

print(data['input_ids'].shape)

print(data['attention_mask'].shape)

print(data['targets'].shape)


# In[28]:


pre_trained_model_ckpt = 'bert-base-uncased'

class GradeClassifier(nn.Module):
    def __init__(self, n_classes):
        super(GradeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_ckpt, return_dict = False)
        self.drop = nn.Dropout(p = 0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


# In[29]:


#Model Utility Class

class_names = ['F-Grade', 'C-Grade', 'B-Grade', 'A-Grade', 'O-Grade']

model = GradeClassifier(len(class_names))

model = model.to(device)

input_ids = data['input_ids'].to(device)

attention_mask = data['attention_mask'].to(device)

F.softmax(model(input_ids, attention_mask), dim = 1)


# In[30]:


#Training

EPOCHS = 10

optimizer = optim.AdamW(model.parameters(), lr = 1e-5)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps=total_steps)

loss_fn = nn.CrossEntropyLoss().to(device)


# In[31]:


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    
    
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        _, preds = torch.max(outputs, dim = 1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets).cpu()
        
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        
        optimizer.step()
        scheduler.step()
        
    return correct_predictions/n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        _, preds = torch.max(outputs, dim = 1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets).cpu()
        
        losses.append(loss.item())
        
    return correct_predictions/n_examples, np.mean(losses)


def get_predictions(model, data_loader):
    all_student_input_ids = []
    model = model.eval()
    student_answer = []
    predictions = []
    prediction_probs = []
    real_values = []
    
    with torch.no_grad():
        for d in data_loader:
            print(d)
            #student_ans = d['student_answers']
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            all_student_input_ids.append(d['student_input_ids'])
                          
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            _, preds = torch.max(outputs, dim = 1)
                          
            probs = F.softmax(outputs, dim = 1)
                          
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)
                          
    student_input_ids = torch.cat(all_student_input_ids, dim=0)                       
    predictions = torch.stack(predictions).cpu()
    predictions_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
                         
    return student_answer, predictions, prediction_probs, real_values


# In[35]:


import time
import pickle

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1} / {EPOCHS}')
    start_time = time.time()  # Track epoch time
    
    train_acc, train_loss = train_model(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))
    
    print(f"Epoch Time: {time.time() - start_time:.2f} seconds")
    
    
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    
    if val_acc >best_accuracy:
        torch.save(model.state_dict(), 'best_model.bin')
        best_accuracy = val_acc
        
f = open("db.model", "wb")
pickle.dump(model, f)
f.close()


# In[36]:


overall_train_loss = history['train_loss'][-1]  # Last element of the training loss list
overall_train_acc = history['train_acc'][-1]  
overall_val_loss = history['val_loss'][-1]  
overall_val_acc = history['val_acc'][-1] 


# In[37]:


print("Overall Results:")
print(f"Final Training Accuracy: {overall_train_acc:.3f}, Final Training Loss: {overall_train_loss:.3f}")
print(f"Final Validation Accuracy: {overall_val_acc:.3f}, Final Validation Loss: {overall_val_loss:.3f}")


# In[99]:


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot = True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation = 0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation = 30, ha='right')
    plt.ylabel('True Grade')
    plt.xlabel('Predicted Grade')
    
y_grade_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns = class_names)
show_confusion_matrix(df_cm)


# In[48]:


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import numpy as np

# Configuration
model_path = 'best_model.bin'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  

# Load the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5) 
state_dict = torch.load(model_path)
model.load_state_dict(state_dict, strict=False)  # Load matching components
  # Assuming you have 'device' set up

# Function to process your input text
def preprocess_input(text):
    """
    This function should tokenize and preprocess your text the same way 
    it was done during fine-tuning. 
    """
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    return inputs


# Preprocess the input
processed_input = preprocess_input(test_input)
input_ids = processed_input['input_ids'].to(device)
attention_mask = processed_input['attention_mask'].to(device)

def map_label_to_grade(label):
    grade_map = {
        0: 'F',
        1: 'C',
        2: 'B',
        3: 'A',
        4: 'O'
    }
    return grade_map.get(label, 'Unknown')  # Handle unexpected labels


# Get the prediction
model.eval()
with torch.no_grad():
  outputs = model(input_ids=input_ids, attention_mask=attention_mask)
  logits = outputs.logits  # Extract the logits
  _, prediction = torch.max(logits, dim=1)  # Find the max

# Assuming your labels are represented as integers 0-4:
predicted_label = prediction.item()  
predicted_grade = map_label_to_grade(predicted_label)

print(f"Predicted label: {predicted_label}")
print(f"Predicted grade: {predicted_grade}")


# In[44]:


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd

data = pd.read_csv('student_grades_updated.csv')

# Configuration
model_path = 'best_model.bin'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict, strict=False)  # Load matching components
model.eval()  # Set model to evaluation mode

# Function to process your input text
def preprocess_input(text):
  """
  This function should tokenize and preprocess your text the same way
  it was done during fine-tuning.
  """
  inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
  return inputs

# Assuming your dataset is a pandas DataFrame named 'data'
def predict_grades(data, num_rows=15):
  """
  This function iterates over the first 'num_rows' of the dataset,
  predicts grades, and prints the results along with actual grades.
  """
  for i in range(min(num_rows, len(data))):
    text = data.iloc[i]['random_answer_list']  # Assuming 'text' column contains the answer
    actual_grade = data.iloc[i]['Grades']  # Assuming 'grade' column contains the actual grade (modify if different)
    processed_input = preprocess_input(text)
    input_ids = processed_input['input_ids'].to(device)
    attention_mask = processed_input['attention_mask'].to(device)

    def map_label_to_grade(label):
      grade_map = {
          0: 'F',
          1: 'C',
          2: 'B',
          3: 'A',
          4: 'O'
      }
      return grade_map.get(label, 'Unknown')  # Handle unexpected labels

    with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      logits = outputs.logits  # Extract the logits
      _, prediction = torch.max(logits, dim=1)  # Find the max

      predicted_label = prediction.item()
      predicted_grade = map_label_to_grade(predicted_label)

      print(f"Row {i+1}:")
      print(f"  Actual Grade: {actual_grade}")
      print(f"  Predicted grade: {predicted_grade}")
      print("-" * 30)  # Optional separator

# Assuming you have your dataset loaded into a pandas DataFrame named 'data'
predict_grades(data)


# In[ ]:




