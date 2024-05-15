# How to Install and Run the Project

## Required folder structure to run files:
```markdown
├── data
│   ├── images         <- Images Folder
│   ├── Masks          <- Masks Folder
│   └── metadata.csv   <- Labels, demographic variables etc
│
├── fyp2024            <- Your project repository
│
```
Note: Images and mask should be of type png


## How To Run:
### Create model:

Markup : * Run "01_process_images.py" to extract the features from all of your images.
         * The features then gets extracted into a csv file on your machine.
         * Then run "02_train_classifiers.py" with the csv file
         * Then save the model by running "03_evaluate_classifier.py"




### Test model

