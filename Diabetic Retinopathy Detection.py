# %% [markdown]
# *Importing libraries*

# %%
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from collections import Counter
from imblearn.over_sampling import SMOTE

# %% [markdown]
# *Data Ingestion*

# %%
#loading all .npz files from the directory
def load_data(directory):
    images, labels, genders = [], [], []
    npz_files= glob.glob(os.path.join(directory, '*.npz'))
    for file in npz_files:
        data= np.load(file)
        images.append(data['slo_fundus'])
        labels.append(data['dr_class'])
        genders.append(data['male'])
    return np.array(images), np.array(labels), np.array(genders)

# %%
#paths to the datasets
train_data_dir= "C:/Users/eutomi/Downloads/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos-20241202T032045Z-001/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/train"
test_data_dir= "C:/Users/eutomi/Downloads/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos-20241202T032045Z-001/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/test"
val_data_dir= "C:/Users/eutomi/Downloads/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos-20241202T032045Z-001/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/val"

# %%
#loading the datasets
train_images, train_labels, train_genders= load_data(train_data_dir)
test_images, test_labels, test_genders= load_data(test_data_dir)
val_images, val_labels, val_genders= load_data(val_data_dir)

# %%
#checking the shape of the datasets
print("Shape of train images: ", train_images.shape)
print("Shape of test images: ", test_images.shape)
print("Shape of validation images: ", val_images.shape)

# %%
#normalizing images
train_images= train_images/255.0
test_images= test_images/255.0
val_images= val_images/255.0

# %%
#checking for class imbalance
data_dir= "C:/Users/eutomi/Downloads/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos-20241202T032045Z-001/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/train"
#getting paths to the .npz files
npz_files= glob.glob(os.path.join(data_dir, '*.npz'))
#checking values for a few sample files
sample_files= npz_files[:5]
for file in sample_files:
    data= np.load(file)
    print(f"file:{file}")
    print("Diabetic Retinopathy Class ('dr_class') value:", data['dr_class'])
    print("Gender ('male') value:", data['male'])

# %%
#initializing counters for 'male' and 'dr_class'
gender_counter= Counter()
label_counter= Counter()

#looping through all .npz to count for occurences of values
for file in npz_files:
    data= np.load(file)
    gender_counter[int(data['male'])] +=1
    label_counter[int(data['dr_class'])] +=1

#print counts for genders and labels
print("Counts for 'male' (gender):", gender_counter)
print("Counts for 'dr_class' (label):", label_counter)

# %%
#calculating class weights to handle imbalance
class_weights= class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict= {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)

# %% [markdown]
# **Modeling 1**

# %%
#model architecture (ResNet50)
def build_model():
    base_model= ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
    model= Sequential(
        [base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')]
    )
    return model

#compiling and training the model
model= build_model()
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['AUC'])

history= model.fit(
    train_images, train_labels, 
    validation_data=(val_images, val_labels),
    epochs=10, 
    batch_size=32, 
    class_weight=class_weight_dict
)
#evaluating the model
results= model.evaluate(test_images, test_labels)
print("Test AUC:", results[1])

# %%
#calculating AUC using sklearn for comparison
test_predictions = model.predict(test_images).ravel()

#overall AUC score
overall_auc= roc_auc_score(test_labels, test_predictions)
print("Overall AUC:", overall_auc)

#separating AUC for Male and Female groups
female_indices= np.where(test_genders ==0)[0] #assuming 0= female, 1= male
male_indices= np.where(test_genders == 1)[0]

female_auc= roc_auc_score(test_labels[female_indices], test_predictions[female_indices])
male_auc= roc_auc_score(test_labels[male_indices], test_predictions[male_indices])

print("Female AUC:", female_auc)
print("Male AUC:", male_auc)

# %% [markdown]
# **Modeling 2**

# %%
#loading and preprocessing the data
def preprocess_data(images, labels):
    images= images/255.0
    return images, labels

#synthetic oversampling using SMOTE
def apply_smote(images, labels):
    #reshape images to 2D for SMOTE
    n_samples, height, width, channels= images.shape
    flat_images= images.reshape(n_samples, -1)

    #applying SMOTE
    smote= SMOTE(random_state=42)
    oversampled_images, oversampled_labels= smote.fit_resample(flat_images, labels)
    
    #reshaping images back to 3D
    oversampled_images= oversampled_images.reshape(-1, height, width, channels)
    return oversampled_images, oversampled_labels

#building EfficientNetB0 model
def build_efficientnet_model():
    base_model= EfficientNetB0(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
    model= Sequential(
        [base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')]
    )
    return model

#loading the datasets
train_images, train_labels, train_genders = load_data("C:/Users/eutomi/Downloads/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos-20241202T032045Z-001/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/train")
val_images, val_labels, val_genders = load_data("C:/Users/eutomi/Downloads/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos-20241202T032045Z-001/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/val")
test_images, test_labels, test_genders = load_data("C:/Users/eutomi/Downloads/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos-20241202T032045Z-001/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/test")

#preprocessing the data
train_images, train_labels= preprocess_data(train_images, train_labels)
val_images, val_labels= preprocess_data(val_images, val_labels)
test_images, test_labels= preprocess_data(test_images, test_labels)

#applying SMOTE to balance the training data
oversampled_images, oversampled_labels = apply_smote(train_images, train_labels)

#building and compiling the model
model= build_efficientnet_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])

#training the model using the validation set
history= model.fit(
    oversampled_images, oversampled_labels,
    validation_data=(val_images, val_labels),
    epochs=10,
    batch_size=16
)

#evluating the model on the test set
results= model.evaluate(test_images, test_labels)
print("Test AUC:", results[1])

# %%
#metrics for the model
test_predictions= model.predict(test_images).ravel()

#overall AUC score
overall_auc= roc_auc_score(test_labels, test_predictions)
print("Overall AUC:", overall_auc)

#gender-based AUC scores
female_indices= np.where(test_genders == 0)[0]
male_indices= np.where(test_genders == 1)[0]

female_auc= roc_auc_score(test_labels[female_indices], test_predictions[female_indices])
male_auc= roc_auc_score(test_labels[male_indices], test_predictions[male_indices])

print("Female AUC:", female_auc)
print("Male AUC:", male_auc)


