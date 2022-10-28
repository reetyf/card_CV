Images are from a public dataset downloadable from Kaggle
Click To download: https://www.kaggle.com/datasets/jaypradipshah/the-complete-playing-card-dataset/download

The database and csv file are not included in the submission due to their large size.

NOTICE: concurrent running of these notebooks concurrently might cause memory depletion issues. It is recommended to run each notebook then close the kernel, then reopen and run the subsequent notebook.


The data can be spawned through the 'load_data_w_EDA.ipynb' file with a load time of around 4 hours. 

The python files 'preprocess.py' and 'helpers.py' should be in the same directory as the notebooks.

The notebooks are to be run in order:
1. load_data_w_EDA.ipynb 
2. log_reg.ipynb
3. random_forest_xg_boost.ipynb
4. deeplearning.ipynb

Be aware that the estimated loading times for the first three notebooks is 7-10 minutes. For the last notebook, loading is done in 2-3 minutes.

Necessary Imports: (instruction: pip install -r requirements.txt)

pandas

numpy

matplotlib

opencv (conda install opencv-python)

seaborn

glob (conda install glob2)

tqdm

elementpath

XGBoost (conda install -c conda-forge xgboost)

sk-learn (conda install scikit-learn)

tensorflow (conda install -c conda-forge tensorflow)
