# Just a Dog Identification Service [JaDIS]
ML + MobileApp = JaDIS

This repo contains an implementation of a tensorflow model which is then compiled into tensorflow lite model and finally used on an android application to identify dog breeds

## Preview
Here are some examples of the final project:

| Some | static                                                                                                                 | screenshots                                                                                                             |
| --- |------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| ![architecture](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/app_example_static_1.jpg) | ![architecture](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/app_example_static_2.jpg) | ![architecture](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/app_example_static_3.jpg ) |
![architecture](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/TODO.png)
![architecture](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/TODO.png)

## The dataset
For training the convolutional neural network I have used the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) which contains around 20k images of 120 different dog breeds.

![all the breed on single image](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/TODO.png)

## Neural network model                 [lupus_omni_die.py]()

For the final project I have trained two different models (as during the project I have trained countless other models on this dataset to see their performance)

- One using transfer learning
- One using Classical CNN architecture

The reason why transfer learning was necessary in final implementation is because of the sheer diversity and size of the dataset, training a reliable network which would have acceptable accuracy would require quote a complex architecture which would take an extremely long time on the hardware I have at my disposal thus I have decided to use Densenet121 pretrained model as part of the final model which is used in the android APP.

### Network architecture(s)

![arch. with transfer learning](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/TODO)
![architecture](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/TODO.png)

### Training summary
![training with transfer learning](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/w_transfer_learning.png)
![training without transfer learning](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/TODO.png)

As we can see by the examples above the transfer learning model is able to achieve way more accurate results as it is more complex model while using relatively similar computation time as the classical CNN model.

### TFLite model summary

Lastly the model is compiled using TFLite compiler into a .tflite file which then is easily accessed by android app. The trained model loses some accuracy due to compression but is still nonetheless accurate as seen in all the previews

#### METADATA INFORMATION OF THE TFLITE MODEL
![TFLite metadata](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/augmentation_example.png)

### Conclusion



## Android App Implementation           [JustADogIdentificationApp](https://github.com/Potentiak/JustADogIdentificationService/tree/main/JustADogIdentificationApp)
The final part of the project is the android implementation which actively uses the trained and compiled model to classify a breed of dog currently in front of the device's camera.
The app consists of one activity which implements CameraX for live image data which is displayed on preview and fed into the TFLite model which den return predictions which are displayed just below the preview window as a recycler view list.
The app only diplays 10 most probable guesses.

![app layout](https://github.com/Potentiak/JustADogIdentificationService/blob/main/figures/app_layout_live.jpg)