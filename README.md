This repository provides the code for [Multimodal transformer-based model for predicting prognosis after radiotherapy plus systemic therapy in hepatocellular carcinoma]. Based on the code, you can easily train your own AURA by configuring your own dataset and modifying the training details (such as optimizer, learning rate, etc).

## Overview

AURA is a transformer-based multimodal medical prediction model that can perform both classification and survival prediction tasks. It processes data from three modalities *[imaging, text, and structured metrics]** and integrates them using cross-attention mechanisms for fusion.*

## Setup the Environment

This software was implemented in a system running Windows 10, with Python 3.9, PyTorch 2.5.1, and CUDA 12.1.

You can adjust the batch size to adapt to your own hardware environment. Personally, we recommend the use of four NVIDIA GPUs.

## Code Description

The main architecture of AURA lies in the models/ folder. The files modeling_aura.py and modeling_aura_surv serve as the main backbone for classification tasks and survival tasks, while the rest necessary modules are distributed into different files based on their own functions, i.e., attention.py, block.py, configs.py, embed.py, encoder.py, and mlp.py. Please refer to each file to acquire more implementation details.

Parameter description:

--CLS: number of classification

--BSZ: batch size.

--DATA_DIR: Folder path for the imaging data. arranged to have portal and arterial subfolers inside.

--SET_TYPE: file name of the clinical baseline data (***.pkl).

Note that xxx.pkl is a dictionary that stores the clinical textual data in the format of key-value. Here is a short piece of code showing how to organize the ***.pkl:

```python
import pickle
f = open('***.pkl', 'rb')
subset = pickle.load(f)
f.close()
list(subset.keys())[0:10] # display top 10 case ids
key = list(subset.keys()) # select the patient ID
subset[key] # display the clinical data
subset[key]['Baseline'] # the demographics information (age and sex)
subset[key]['Lab'] #  the laboratory test results
subset[key]['label'] # the clinical endpoint labels
```

