[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11110470&assignment_repo_type=AssignmentRepo)
# XNAP-Project title (replace it by the title of your project)
Write here a short summary about your project. The text must include a short introduction and the targeted goals
- main objective: classify the different patches in 0 or 1 depending whther the image is part of a malignous tumour or not.
- data: different folders with the patients' id, inside each folder, two folders, 0 and 1 depending if it is a bennignous tumour or malignous
- it has been implemented different models trying to reach best performance possible in order to ensure a good model quality for detecting tumours in "mamografias"
- cnn: select this type of nn since it is (definition, images, classification...) + own architecture
          - cross entropy loss: because ...
          - focal loss: diff amount of data (0 and 1) 
          - bcelogits: binary, class ...
- resnet: try with pretrained model and adjust them for our data and objective 
- mobilenet + finetuning: try with a different pretrained model and added a finetuning for (..)

## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.


## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```



## Contributors
Write here the name and UAB mail of the group members
Paula Feliu Criado, p.feliu12@gmail.com
Roger Garcia, 
Montserrat Farres,

Xarxes Neuronals i Aprenentatge Profund
Grau de Intel·ligència Artificial, 
UAB, 2023
