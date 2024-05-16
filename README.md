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

Note: Masks must also contain same name as image but need "_mask" at the end


## How To Run:
### Create model:
#### Instructions for Processing Images and Training Classifiers

1. **Extract Features from Images:**
   - Run `01_process_images.py` to extract features from your images.
   - The extracted features will be saved into a CSV file on your local machine.

2. **Train Classifiers:**
   - After extracting features, execute `02_train_classifiers.py` with the CSV file generated in the previous step.
   - If file structure is correct `02_train_classifiers.py` will locate the CSV file automatically.

3. **Save the Trained Model:**
   - To save the trained classifier model, run `03_evaluate_classifier.py`.

**Note:** Make sure to run these scripts sequentially in the given order for successful execution.


### Test model
1. **Import Classifier File:**
   - Import classify() function from `03_evaluate_classifier.py` into your testfile

2. **Predict Test Images:**
   - Run the `classify(image)` function with args: image file name
   - If folder structure is correct, image/mask folder will automatically be found


