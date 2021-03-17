#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.io import loadmat

onsets = loadmat("/Users/lovira/Desktop/402Folder/Main/Data/Stimuli/all_katerina_onsets.mat")['onsets']
path_to_text = "/Users/lovira/Desktop/402Folder/Main/Data/Stimuli"
story_files = [f for f in os.listdir(path_to_text) if f.endswith('csv')]
df = pd.read_csv(os.path.join(path_to_text, story_files[0]))

#load eeg data

#data dir
data_dir = "/Users/lovira/Desktop/402Folder/Main/Data/EEG"
eeg_file = [f for f in os.listdir(data_dir) if f.endswith('set')]

# Load data to memory
raw = mne.io.read_raw_eeglab(os.path.join(data_dir, eeg_file[0]), preload=True) #Story 1


tags = [nltk.pos_tag([w])[0][1] for w in df.word]
df['POS'] = tags
#df.head()

raw = raw.filter(1, 10, n_jobs=-1)


# In[19]:


raw = raw.set_eeg_reference() #to apply average reference


# In[43]:


samples_NN = raw.time_as_index(df.loc[df.POS =='NN', 'onset'].values + onsets[0, 0]) # samples for 1 one story, one subject,m just nouns
id_column_NN = np.ones_like(samples_NN)
duration_column_NN = np.zeros_like(samples_NN)

df.loc[df.POS=='NN', 'onset'].values + onsets[0, 0]


# In[100]:


samples_JJ = raw.time_as_index(df.loc[df.POS == 'JJ', 'onset'].values + onsets [0,0]) # Samples for Adj, story 1
id_column_JJ = np.full_like(samples_JJ, 2)
duration_column_JJ = np.zeros_like(samples_JJ)

events_NN = np.array([samples_NN, duration_column_NN, id_column_NN])
events_JJ= np.array([samples_JJ,duration_column_JJ, id_column_JJ])

events = np.hstack((events_NN, events_JJ))
                 
event_id = {'noun': 1, 'adjective': 2}


# In[94]:


# Epochs and Averaging
epochs = mne.Epochs(raw, events, event_id)

evoked = epochs.average()

raw = raw.set_eeg_reference() #to apply average reference


# In[93]:


#Plots
evoked.plot_joint(times=[0.05, 0.250])


# In[98]:


evoked.plot()


# In[ ]:





# In[ ]:





# In[ ]:




