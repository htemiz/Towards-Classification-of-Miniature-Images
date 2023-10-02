# Towards-Classification-of-Miniature-Images
Repository for supporting files and outcomes for my paper entitled
___[Towards Accurate Classification of Miniature Images](https://www.amerikakongresi.org/_files/ugd/797a84_42d94c1e33d641d4a0615d9494ee582c.pdf)___
presented in 
___[Latin America 5th International Conference on Scientific Researches](https://www.amerikakongresi.org/)___, March 17-19, Medellin, Columbia, 2023.


Please cite this proceeding as follows:


*Temiz, H. (2023). Towards Accurate Classification of Miniature Images. Latin America 5th International Conference on 
Scientific Researches,Medellin, pp. 181-187.*



Please feel free to contact me at [htemiz@artvin.edu.tr](mailto:htemiz@artvin.edu.tr) for further information and comments.

### Overview
Miniatures are small images used to drawn on manuscripts to visually describe the subject of the manuscripts. 
Miniature paintings are made to provide a better understanding of what is told in the texts or to strengthen 
the narration. Ottomans used the miniatures between 14th and 18th centuries.

They depict portraits, lives of sultans, festivals, historical events, life style, nature and city views, 
literary works, religious subjects, traditions and customs, women and men, and creatures such as animals and plants.

In this work aimed to teach computer to identify the artists of the miniatures from given images. To accomplish this, 
a convolutional neural network (CNN) is trained with some miniature images of four different artists. 

### Dataset
Dataset consists of 380 images belonging the following four artists: 
Levni, Matrakçı Nasuh, Rumuzi and Seyyid Lokman. The images were downloaded from [https://www.turkishculture.org](https://www.turkishculture.org)

### Algorithms
Entire experiment is done with Keras. The architecture of the algorithms as follows:

[DECUSR](models%2FDecusr_4RB.py)

[REDNet](models%2FREDNET.py)

[VDSR-19](models%2FVDSR.py)

[SRCNN](models%2FSRCNN.py)


Each model files is a type of DeepSR file. To train, e.g., DECUSR, issue the following command:


```shell
python -m DeepSR.DeepSR --modelfile models/Decusr_4RB.py --train
```

Please specify the path of training directory by adding ```--traindir <path to training directory>``` command parameter.
It can also be specified with 'traindir' keyword in the 'settings' dictionary located in the model file.

To test the model, add ```--test --testpath <path to your test files>``` command arguments. Please remove ```--train```
if you will not train and just test the model. 

TO re-initiate the model with pre-trained weights, add ```--weightpath <path to your weight file>``` 

The parameters can , as some of them are illustrated just above lines, be set in the 'settings' dictionary in the 
model files. 

For additional information and examples, please refer to the documentation of DeepSR:

    [https://github.com/htemiz/DeepSR](https://github.com/htemiz/DeepSR)



[](images/cnn.png)

### Training








