# twitter-chainsaw
Fundamentals of ml group project. Twitter data is used to predict stocks. 
<br/>
<br/>

# Files and their purposes

The word model can be trained and run through the "train_word_model.py" file.
* Constants are set just under the entry point (if __name__=="__main__").
* The "epochs" variable determines how many epochs the model is trained on for each fold of cross validation.
* The "use_word_mat" variable is True or False, and determines whether the WordProp algorithm is used.
* The "categorical" variable is True or False, and determines whether the stocks being predicted are categorical.
<br/>
"train_stock_forecaster.py" is code used for running a model that predicts future stocks based on previous stocks.

The "wordprop" folder is code relevant to the use of the novel approach.

The "glove.6B/" folder contains files used for the glove model, which is used to calculate word similarity.
