
# Respiratory Disease Classification


## Project status


## Introduction
The breathing sound test is a telemedical online lung auscultation that automatically returns the probability of having healthy airways. A result <70% is an indication for consulting a physician. In case of risk factors or symptoms, further testing is recommended at even higher probabilities.

## Table of contents
* [Motivation](#Motivation)
* [Software build](#Software-build)
  * [Build with Python](#Build-with-Python)
  * [Build with JavaScript](#Build-with-JavaScript)
  * [Build with Streamlit](#Build-with-Streamlit)
* [Training data](#Training-data)

## Motivation
[Seven percent of the world population](https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(20)30157-0/fulltext) suffer from chronic respiratory diseases, mainly COPD. 3.9 million persons died from them in 2017, and even more became disabled. Before COVID19, the one-year global incidence of acute respiratory diseases was already near to 100%. COPD is the third and lower respiratory infections the fourth [global leading cause of death](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death).

Telehealth reduces the burden on medical resources, saves the patients´ time and money, and is easily accessible regardless of the location. Telemedical care [increased during the COVID-19 pandemic](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7532385/). The [evolution of convolutional neural networks](https://ieeexplore.ieee.org/document/9156454) further contribute to this increase.
## Training and validation datasets
The [ICBHI dataset](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) and the [Steth dataset](https://data.mendeley.com/datasets/jwyy9np4gv/3) include a total of 238 participants (37% girls/women) from the infancy up to >90 years of age. 177 participants suffered respiratory diseases, which were chronic in 124 cases, predominately COPD (75 cases) and asthma (34 cases). The participants provided 25618 seconds of respiratory auscultation, namely 2570 seconds by the healthy and 23048 seconds by the respiratory ill. The duration of the respiratory cycles was - on the average of the participant averages of the ICBHI set - 2.4 in the healthy and 2.2 in the acutely ill versus 3.0 in the chronically ill.
## Programming
The most important programming steps are a deep convolutional neural network and a client app.
### Preprocessing and augmentation
The units for training the network are figures of constant size obtained from the wav audio files:
- slicing to eight-second audio chunks with 10% overlap for sounds form health persons and 90% overlap for sounds from persons with respiratory diseases (to balance the dataset)
- zero-pre-padding of residual pieces, split-up of noises into volumes by Fourier transformation, and conversion into spectrograms
- data augmentation with random volume reduction (darkening), random frequency masking (horizontal bars of zeros), and random time frame masking (vertical bars of zeros)
### Deep convolutional neural network
Transfer learning with ResNet50V2 pre-trained on the [ImageNet Classification Problem](https://keras.io/api/applications/resnet/#resnet50v2-function):
- Tensor flow python module
- 47 convolutional layers, a max pooling layer, an average pooling layer, and a fully connected output layer with 1000 nodes and softmax activation function
- replacement of the original output layer by a densely-connected output layer with 2048 weights, sigmoid activation function, and one final output
- first freezing of the original layers and training of the new output layer, then retraining of the entire network with a 40 times smaller initial learning rate
### Client app
One app is built with Streamlit, the other app is built with HTML, CSS, and JavaScript. They include:
- during the initial state:
  - a train button with instructions to improve the quality
  - a more button leading to this readme file
  - a contact button to the members of this project
  - a scroll menu to collect more heterogenous train data
  - a button to start the 16-second recording phase
- during the return state:
  - the probability of having health airways with and without smoking
  - a threshold below which consulting a physician is recommended
  - a table with possibilities to further improve the recording quality
## Demo
### [Full web application](https://medscoops.com/capstone)
### [Streamlit application](https://medscoops.com/capstone)

## Run Locally

Clone the project

```bash
  git clone git@github.com:loukra/Respiratory_Disease_Classification.git
```

Go to the project directory

```bash
  cd Respiratory_Disease_Classification
```
Create Virtual Envirnment 

```bash
  pyenv local 3.9.8
  python -m venv .venv
  source .venv/bin/activate
```

Install dependencies

```bash
  pip install --upgrade pip
  pip install -r requirements.txt
```

## Tech Stack

**Client:** ...

**Server:** ...


## Authors

- [@loukra](https://www.github.com/loukra)
- [@Li](https://www.github.com/loukra)
- [@Rafael](https://www.github.com/loukra)

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherineoelsner.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/)


## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Appendix

Any additional information goes here

