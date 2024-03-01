Medical imaging report

Introduction
As part of our first project in data science we have been provided with medical images containing skin lesions. The purpose of this report is to give a brief overview of the content of the images by examining what diagnoses are present in the photos, investigating what data is available, and the quality of the content.

Data overview
As part of the project our group has been provided with part of the public data set PAD-UFES-20. Our part of the data set contains 127 images each containing an image skin with a lesion on it. The images are in PNG format and ranges anywhere from 85 kilobytes to 10 megabytes. The photos are taken up close and some of them contain a circle or an arrow drawn on the skin to indicate where the lesion is present.

Types of diagnosis
Through thorough examination we have come to discover a range of diagnosis present in the images. Despite limited medical knowledge we have tried to go through the photos and compare them to different images found in medical journals on google scholar and dermnetnz. Below are our own images, and underneath them contain a description of our suggestion as to what the diagnosis of the image may be. 


 
PAT_625_1184_412

Source: Dermnet	

This lesion appears to be a raised nodule with a mix of pigmentation, including pink and darker areas that may suggest melanocytic activity.

 
PAT_629_1192_362	 

Source: Dermnet	

This lesion shows a lesion with an irregular border and colors ranging from red to purple with crusting. A diagnosis could be psoriasis or eczema.

 
PAT_680_1289_585
	 
Source: Dermnet	This appears to be a well-demarcated, pigmented lesion with an irregular outline that could be suggestive of a nevus.

PAT_787_1482_328

Source: Dermnet

This lesion has a black center with a circular brown spots. It has multiple parts in a specific area of the skin. It can be diagnosed as Cutaneous squamous cell carcinoma (SCC) which usually begins as a tender erythematous papule or nodule that may have a variable amount of hyperkeratosis. Older lesions may ulcerate.

These images of lesion and their descriptions highlights a few of our findings among a longer list of possible diagnosis in the dataset we have been provided with including melanocytic, melanoma, squamous cell carcinoma, Basal cell carcinoma, psoriasis, eczema, and dermatofibroma. 

General information
Skin lesions can be hereditary, for example moles or birthmarks. However, the most common cause of skin lesions is infection, such as herpes simplex, human immunodeficiency virus (HIV) or human papillomarvirus (HPV). Skin lesions can also occur due to allergic reactions, including reaction to medication such as corticosteroids or chemotherapy. Another way skin lesions can form is through external ways, such as sun exposure, insect bites, burns, circulation issues and vitamin deficiencies. However, along with these easily treatable lesions, lesiosn can also form due to systemic diseases, like autoimmune diseases, liver and kidney disease, some infectious diseases and of course cancer.

Observations and conclusion
After inspecting the images we have annotated them by coloring the lesion in label studio. Some of the images contained lesions that were difficult to annotate which made us skip them and focus on the ones where the lesions were clear and easy to annotate. However, this still resulted in over 100 images of lesion.

References 
Moles (melanocytic naevi, pigmented nevi) | DermNet (dermnetnz.org)
Erythrasma | DermNet (dermnetnz.org)
Dermatitis: Types and treatments — DermNet (dermnetnz.org)
Psoriasis: Symptoms, Treatment, Images and More - DermNet (dermnetnz.org)
A machine learning approach for skin disease detection and classification using image segmentation - ScienceDirect






