# MWTCNN


#### 🌟 Time Frequency Distribution and Deep Neural Network for Automated Identification of Insomnia using Single Channel EEG Signals 🌙
Overview

Welcome to our sleep revolution repository! Here, we harness the power of time-frequency distribution techniques and deep neural networks to automate the identification of insomnia using single-channel EEG signals. Sweet dreams are made of this!

🚀 Getting Started

![InsomniaStudy](https://github.com/prince0310/MWTCNN/assets/85225054/f2711131-23ce-4c72-903c-847bc15fe873)

#### 📊 Data acquisition 

In this paper, two publicly available databases: CAP - sleep and (ii) SDRC have been used for the experimentation

#### ✨ Data pre-processing and segmentation

The databases recordcontains recording of various signals
through polysomnography (PSG), including respiratory, EOG,
EEG, and ECG signals. In our experiment, we focused on
extracting EEG signals specifically from the C4 − A1 channel
in the PSG recordings. To reduce the processing time and
increase the accuracy of detection, we have segmented the
whole signal into 1 sec. duration and converted these sub-
signals into RGB images using continuous wavelet transform
with Morlet mother wavelet fun. <br> <br>


![EEG-recording-1sec-CAP](https://github.com/prince0310/MWTCNN/assets/85225054/e1b37205-2f4a-4d19-aff4-7c64e75f79aa)


#### 📈 Scalogram generation

The scalogram of the signal is extracted using Morlet
CWT (Continuous wavelet transform). The deep convolutional
neural network component of the proposed methodology uses
CWT coefficients as features. This work evaluates the per-
formance of the proposed method on Morlet wavelet func-
tions and Smoothed pseudo wigner-wille distribution based
scalograms. Calculating the coefficients scales is done on the
1sec segment of each signal. Then scalograms are retrieved
and scaled using bi-cubic interpolation following the specifica-
tions.<br> <br>

![scalograms-all](https://github.com/prince0310/MWTCNN/assets/85225054/8b42c178-9878-43ea-9b17-701ce40aba01)


#### 🛠️ Transfer Learning

Large annotated dataset, time, and high computing resources
are required for training a CNN from scratch than using
a CNN that has been pre-trained on a huge database [26].
There are two primary transfer learning scenarios: freezing
and fine-tuning the layers of CNN architecture. In fine-tuning,
the weights and biases of a CNN that has already been
trained are used instead of random initialization, followed by
a normal training procedure on the unseen dataset. However,
in another scenario, the pre-trained CNN layers are considered
to be fixed feature extractors. In this case, the fully connected
layers are tweaked across the target dataset and number of
defined classes whereas the biases and weights of our ideal
convolutional layers remain fixed. The frozen layers are not
restricted to convolutional layers alone. Frozen layers may
be fully connected layers or any subset of convolutional;
nevertheless, it is usual practice to freeze the more superficial
convolutional layers. In our study, we have used the freezing
method by freezing all the CNN layers and re-trained the last
three fully-connected layers with number of output classes
as two for last layer of AlexNet , GoogLeNet,
VGG16, ResNet50, and MobileNetV2, using
the same training pipeline used for the proposed method and
compared the performance of these models with our proposed
MWTCNNet model for classification between normal and
insomniac scalogram

#### 🎲 Clone the repository and cast the spell of installation:

```
git clone https://github.com/yourusername/insomnia-identification.git
cd insomnia-identification
pip install -r requirements.txt
```

🚀 Note -- edit config.yaml file to set parameters before start training

🚀 Note -- edit dataset.py file to provide data path of custom dataset before start training

✨ 🧠  🎲  📈 start training
 ```python3 main.py --model=googlenet/alexnet/resnet --save_model=True```
 
#### ✨ Usage

* Follow the enchanting steps in the Usage document to wield the power of insomnia identification.
🌌 Data Galaxy

* Dive into the vast universe of our dataset in the Data section. Download the cosmic knowledge and unveil the secrets within.
🛠️ Preprocessing Magic

* Discover the magical transformations applied to the EEG signals in the Preprocessing documentation.
⏳ Time-Frequency Alchemy

* Learn the ancient art of time-frequency distribution techniques in the Time-Frequency Distribution section.
🧙‍♂️ Deep Neural Sorcery

* Master the mystical architecture and training process of the deep neural network in the Deep Neural Network documentation.


#### 🎉 Results Celebration

📈 Accuracy curve <br>

![acc_cirves](https://github.com/prince0310/MWTCNN/assets/85225054/de75ae14-816d-42b8-a497-9f9f885870c7)

<br> <br> 

📊  Confusion matrix <br>
![Confusionmatrix(1)](https://github.com/prince0310/MWTCNN/assets/85225054/1d8798d8-e9b1-435f-a33a-13886648de81)



* Join the celebration of the results obtained from the system in the Results section. Cheers to accurate insomnia classification!
🤝 Contributing

* This project is licensed under the [Your License] - see the LICENSE.md file for details.
🙏 Acknowledgments

Special thanks to [contributors or libraries] for their invaluable contributions. Your magic has made this dream come true.

Dream on and feel free to reach out with any questions or concerns. 🌌💤✨
