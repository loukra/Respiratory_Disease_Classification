
# Respiratory Disease Classification - Readme

## Introduction
The breathing sound test is a telemedical online lung auscultation that automatically returns the probability of having healthy airways with and without smoking, along with the caveat that more risk factors and symptoms may further diminish it. With a threshold of 52%, 95% of the respiratory ill are classified as such.

## Table of contents
* [Motivation](#Motivation)
* [Project status](#Project-status)
* [Training and validation data](Training-and-validation-datasets)
* [Programming](#Programming)
  * [Preprocessing and augmentation](#Preprocessing-and-augmentation)
  * [Deep convolutional neural network](#Deep-convolutional-neural-network)
  * [Client applications](#Client-applications)
* [Server/cloud](#Server/cloud)
* [Confidentiality](#Confidentiality)
* [Demo](#Demo)
  * [App1 (full web application)](App1)
  * [App2 (Streamlit application)](App2)
  * [App3 (mobile phone application)](App3)
* [Local editing](Local-editing)
* [Authors](Authors)
* [Links](Links)
* [Acknowledgements](Acknowledgements)

## Motivation
[Seven percent of the humanity](https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(20)30157-0/fulltext) suffer from chronic respiratory diseases, mainly COPD. 3.9 million persons died from them in 2017, and even more became disabled. Before COVID19, the one-year global incidence of acute respiratory diseases was already near to 100%. COPD is the third and lower respiratory infections the fourth [global leading cause of death](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death).

Telehealth reduces the burden on medical resources, saves the patients´ time and money, and is easily accessible. Telemedical care [increased during the COVID-19 pandemic](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7532385/). The [evolution of convolutional neural networks](https://ieeexplore.ieee.org/document/9156454) further contributes to this increase.
## Project status

|<span style='display:inline-block;width:600px;'>Training and validation data</span>||
|---|---|
|[ICBHI data](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)|&check;|
|[Steth data](https://data.mendeley.com/datasets/jwyy9np4gv/3)|&check;|
|[HF_Lung_V1 data](https://gitlab.com/techsupportHF/HF_Lung_V1)||
|Microphone data||
|Training data from the apps||

|Network programming|&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; ||
|---|---|---|
|Model1:|- distinction between healthy and suspicious|&check;|
||- area under the receiver operating characteristic curve||
|Model2:|- is acute or chronic more probable?||
|Model3:|- most probable diagnosis||

|Performance of model1 given a recall of 95% for the respiratory ill                 ||
|---|---|
|Threshold for the predicted probability|52%|
|Recall for the respiratory healthy|67%|
|[Accuracy level reached (0 to 4)](https://www.nature.com/articles/s41598-021-96724-7/figures/4)|3|

|Client app programming &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; |App1|App2|App3|
|---|---|---|---|
|Main functionality|&check;|&check;||
|Full functionality|&check;|||
|Table for further improvement|&check;|||
|Communication with the model||&check;||
|More factors than respiration and smoking||||
## Training and validation data
The ICBHI and Steth datasets include a total of 238 participants (37% girls/women) from the infancy up to >90 years of age. 177 suffered from respiratory diseases, of which 124 were chronic, predominately COPD (75 cases) and asthma (34 cases). The participants provided 25618 seconds of respiratory auscultation, namely 2570 seconds by the respiratory healthy and 23048 seconds by the respiratory ill. The duration of the breath cycles was - on the average of the participant averages of the ICBHI set - 2.4 in the healthy and 2.2 in the acutely ill versus 3.0 in the chronically ill.
## Programming
The most important programming steps are a deep convolutional neural network and client apps.
### Preprocessing and augmentation
The units for training the network are equally large images obtained from the wav audio files as follows:
- slicing to eight-second audio chunks with 90% overlap for sounds from respiratory healthy and 10% for such from respiratory ill persons; i.e. data from the healthy is augmented more than that from the diseased to increase its proportion to nearly 50%
- zero-pre-padding of residual pieces, split-up of noises into volumes by Fourier transformation, and conversion into spectrograms
- further augmentation with random volume reduction (darkening), random frequency masking (horizontal bars of zeros), and random time frame masking (vertical bars of zeros)
### Deep convolutional neural network
Transfer learning with ResNet50V2 pre-trained on the [ImageNet Classification Problem](https://keras.io/api/applications/resnet/#resnet50v2-function):
- Tensor flow python module
- 47 convolutional layers, a max pooling layer, an average pooling layer, and a fully connected output layer with 1000 nodes and softmax activation function
- replacement of the original output layer by a densely-connected output layer with 2048 weights, sigmoid activation function, and one final output
- first freezing of the original layers and training of the new output layer, then retraining of the entire network with a 40 times smaller initial learning rate
### Client apps
App1 is built with HTML, CSS, and JavaScript, app2 with [Streamlit](https://streamlit.io), app3 will run locally on mobile phones. They include:
- during the initial state:
  - a train button with instructions to improve the quality
  - a more button to this readme file
  - a contact button to the members of this project
  - a scroll menu to collect more heterogenous training data
  - a button to start the 16-second recording phase
- during the return state:
  - the probability of having health airways with and without smoking
  - thresholds below which consulting a physician is recommended respectively urgent (traffic light system)
  - a copy of the record for the user
  - a table with possibilities to further improve the recording quality
## Server/cloud
Currently Streamlit and [GitHub](https://github.com/loukra/RespiratoryApp)
## Confidentiality
The additional training data collected within this project is anonymous. Audio with other content than breath cycles, such as voices, etc., are immediately deleted. Users have the possibility to indicate if they
- suspect an airway illness.
- have sound airways.
- have sick airways.

In the latter case, they can further check one of these options:
- any respiratory illness.
- a diagnosis out of a list of of the most common acute and chronic respiratory diseases.

That´s all. The function to collect data will be implemented later.

## Demo
### [App1 (full web application)](https://medscoops.com/capstone)
### [App2 (Streamlit application)](https://loukra-respiratoryapp-streamlit-app-deploy-qhc6bz.streamlit.app)
### App3 (Mobile phone application - to do)
## Local editing

Clone the project

```
git clone git@github.com:loukra/Respiratory_Disease_Classification.git
```

Go to the project directory

```
cd Respiratory_Disease_Classification
```
Create Virtual Envirnment 

```
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```
## Authors

- [@Louis Krause](https://www.github.com/loukra)
- [@Li Xie](https://www.github.com/loukra)
- [@Dr. med. Rafael Cámara](https://www.github.com/loukra)
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherineoelsner.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/)
## Acknowledgements
- [Neue Fische data science coaches: know how](https://www.neuefische.de/team)
- [Arduino Project Hub: idea](https://create.arduino.cc/projecthub/mixpose/digital-stethoscope-ai-1e0229)
- [Project Coswara: skin and structure](https://coswara.iisc.ac.in/?locale=en-US)
- This readme
  - [How to write a good README: know how](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
  - [Awesome README: inspiration](https://github.com/aregtech/areg-sdk#readme)
## Appendix

