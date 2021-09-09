
# Predicting future financial Inclusion in Indonesia

This is a project to predict future ownership of financial instruments. The motivation behind this project was to understand if *future* (not current) ownership of financial instruments (in this case, 7 years in the future), can be reliably predicted using machine learning. 

This has obvious advantages: governments/ NGOs can use it for delivering targeted financial literacy programmes without having to wait for the latest data which takes years to process. And firms that plan to expand their geographical footprint in new territories can predict the level financial education they need to provide to employees-to-be, thus gaining more clarity on costs involved before expanding.

## The data

The dataset is taken from the [Indonesian Family Life Survey](https://www.rand.org/well-being/social-and-behavioral-policy/data/FLS/IFLS/download.html) conducted by RAND, and consists of 10,918 households (rows) after cleaning.

The data consists of 31 columns describing ownership of different assets, annual income per household member, housing type, housing material, and consumption. All the predictor variables (features) are from year 2007.

The target variable is own_fininstrument_14 - which represents ownership of basic financial instruments in 2014. The data is imbalanced. Roughly 71% of households do not own financial instruments ("3"), whereas the rest do ("1").

For dummy variables (such as the asset-ownership columns), "1" = Yes; "3" = No.

The cleaned and ready-to-use dataset is saved in csv format: **ifls_hh_reduced.csv**. This can be reproduced using the ipynb script.

## Running the models

The notebook with codes to clean the data, and run the models is called **fininclusion - dataprep.ipynb**.

Models are tested using imbalanced data, undersampled data, and oversampled data (SMOTE). Using the imbalanced data delivers imprecise predictions, whereas both undersampled and oversampled data have slightly better precision, but poorer recall.


## Streamlit

A simple interactive web-app is built and publicly deployed on Streamlit. The app allows users to explore the data, and build their own machine learning models by selecting a classifier and fine-tuning the hyper parameters. The app is accessible at [bit.ly/joeyyap-financial-inclusion](bit.ly/joeyyap-financial-inclusion).

The file for this is called **capstone.py**. 

## Visualisations
Basic data exploration is conducted in R. The R project, along with visualisations (png) for this are located in the **visualisations** folder. As the main objective (at the time of the project) was to explore differences between the banked and unbanked, feature columns are disaggregated by financial inclusion status.

## Contributing
At the moment, as this project is not being actively maintained, please do not contribute. However, please feel free to use the project for research or personal purposes.

## Author
Jo-yee Yap

## License
[MIT](https://choosealicense.com/licenses/mit/)
