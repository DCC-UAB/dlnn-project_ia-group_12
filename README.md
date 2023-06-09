[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/wT71nrpQ)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11110470&assignment_repo_type=AssignmentRepo)
#  Breast cancer detection
### Data:
This report summarizes the deep learning project focused on classifying mammography images as either benign (0) or malignant (1) tumors. The dataset consists of many patches of images (complete mamographies) from different patients, organized into folders. Each patient's folder contains two subfolders: '0' for images without a malignant tumor and '1' for images with a malignant tumor. The primary objective of the project was to develop models using deep learning techniques, including convolutional neural networks (CNN), pretrained models (ResNet, MobileNet), and a transformer, to accurately classify the images.

The data used to reproduce the code can be downloaded from here: 
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

Another important fact about the data, also mentioned on the notebook, is that the three pre-tained models that have been used, work on the same data but separated in three different folders for train, val and test. This three folders can be downloaded in the following link: https://drive.google.com/drive/folders/17aWK_v2W2uwy6AiduP83HRYlc1_ojur-?usp=sharing


### Introduction:
The aim of this project was to classify mammography images based on the presence or absence of a malignant tumor. To achieve this objective, various deep learning models were implemented, including a CNN, pretrained models, and a transformer. Modifications were made to the models' parameters and architecture to improve their performance.

### Methods:
The initial model considered for achieving the project goal was a CNN, which is suitable for processing images and extracting relevant features. After designing the CNN architecture, it was trained with default values for the optimizer (Adam) and loss function (cross-entropy loss), achieving an accuracy of 87%. To further improve the results, focal loss was employed as the loss function due to the dataset's class imbalance, resulting in an accuracy of nearly 89%. Additionally, the model was tested using the binary cross-entropy loss with logits function, which yielded similar results. The dataset required for executing these models consists of all the patient folders, each containing subfolders '0' and '1' with respective images.

In addition to the CNN with a custom architecture, pretrained models were employed. First, ResNet18 was used due to its effectiveness in image classification tasks. Furthermore, fine-tuning was performed on MobileNet, considering its potential for achieving good results. These pretrained models were trained using 30% of the total available images, and they exhibited satisfactory performance, achieving an accuracy of around 90%.

Finally, a pre-trained transformer model was implemented to classify the images as benign (0) or malignant (1). A Vision Transformer (ViT) model was chosen because it is a more complex model, allowing for a performance comparison with the previously mentioned models. This model is suitable for our objective as it applies attention, has proven to be highly effective in classification tasks with large datasets like ours, and can handle small patch sizes in the images. Additionally, it captures both local and global details, making it a promising option for mammograms, which can have distinctive characteristics. The ViT model can provide generalization of relevant patterns and features.

To execute the pretrained models and the transformer correctly, three additional folders named 'train,' 'val,' and 'test' were included in the repository. The 'train' folder contains 70% of the images, while 'val' and 'test' contain 15% each. These folders also contain subfolders '0' and '1' with images corresponding to benign and malignant tumors, respectively.

### Qualitative Analysis:
Various visualization functions were developed to display the predictions made by the models in a more visually interpretable manner. These functions showcased the original mammography image, highlighting the malignant tumor region with a rosier color, and providing a binary visualization of the tumor's location. These visualizations were generated for all the models and their respective predictions, but only used in some of them as wasn't necessary to collapse the notebook with many equal outputs.

### Conclusions:
In conclusion, this project successfully applied deep learning techniques to classify mammography images as either benign or malignant tumors. The CNN, pretrained models (ResNet and MobileNet), and the transformer model all demonstrated promising results, with accuracies ranging from 87% to 90%. The choice of loss function, such as focal loss, proved effective in addressing class imbalance. Pretrained models offered competitive performance, and the transformer model was a viable alternative, showcasing its potential for complex tasks. The qualitative analysis provided visual interpretations of the models' predictions, aiding in understanding the detection process.

In summary, the project achieved its objectives and presented a comprehensive evaluation of various deep learning models for mammography image classification, paving the way for potential advancements in tumor detection in medical imaging.


## Code structure
The code has been structured in the following way:
- 1 Models folder, in this folder are stored the different models created. In our case there are only two different CNNs and the class of focal loss. It is also important to be pointed that in the execution some pre-trained models have been used that are not defined in this folder.

- 2 Utils folder, in this second folder all different functions used in the code are defined in. Is a way to mantain the code clear and having all functions in the same space. They are also divided into small subgroups depending on the purpose of each function, they are separated by comments especifying the type of functions that you will find afterwards.

- 3 ModelsExecution python notebook, this last file has the complete execution of the differentmodels and the explanations on the decision that have been taken along this project as well as evaluations.


## Example Code
Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate BreastCancerdetection_g12
```



## Contributors
Write here the name and UAB mail of the group members
Paula Feliu Criado, 1630423@uab.cat
Roger Garcia, 1633372@uab.cat
Montserrat Farres, 1636040@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Intel·ligència Artificial, 
UAB, 2023
