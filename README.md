
# Respiratory Disease Classification


## Project status


## Introduction
The breathing sound test is a telemedical online lung auscultation that automatically returns the probability of having healthy airways. A probability <70% is an indication for consulting a physician. In case of risk factors or symptoms, further testing is recommended at even higher probabilities.

## Table of contents
* [Motivation](#Motivation)
* [Software build](#Software-build)
  * [Build with Python](#Build-with-Python)
  * [Build with JavaScript](#Build-with-JavaScript)
  * [Build with Streamlit](#Build-with-Streamlit)
* [Training data](#Training-data)

## Motivation
[Seven percent of the world population](https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(20)30157-0/fulltext) suffer from chronic respiratory diseases, predominately COPD. In 2017, they caused 3.9 million deaths and even more disabilities. Before COVID19, the one-year global incidence of acute respiratory diseases was already near to 100%. COPD is the third and lower respiratory infections the fourth [global leading cause of death](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death).

Telehealth reduces the burden on medical resources, saves the patients´ time and money, and is easily accessible regardless of the location. Telemedical care [increased during the COVID-19 pandemic](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7532385/). The [evolution of convolutional neural networks](https://ieeexplore.ieee.org/document/9156454) further contribute to this increase.

## Software build
### Build with Python
#### Preprocessing and Augmentation
#### Convolutional Neural Network
#### Supported Vector Machine
### Build with HTML, CSS, and JavaScript
#### HTML and CSS
The full web application includes:
- during the initial state:
  - a train button with instructions for a better auscultation, an animation and a respiratory audio sample
  - a more button leading to this readme file
  - a contact button leading to the GitHub accounts of the members of this project
  - a scroll menu to indicate if users are uncertain about their health status, certain to be health or know about a respiratory disease
  - a button to start the recording
- during the return state after the 16-seconds recording phase
  - the probability of having health airways with and without smoking
  - a threshold below which consulting a physician is recommended
  - a table with possibilities to further improve the recording quality
#### Javascript
In addition to run the functions just listed, the script
  - gets access to the users´s microphone
  - updates the scroll menu to offer further options if the users state having diseased airways
  
### Build with Streamlit

### Training data
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

