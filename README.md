## TWENTY NEWSGROUPS CLASSIFER USING SENTENCE TRANSFORMER


In this repository, we use one shot classification to classify the ```fetch_20newsgroups``` of Scikit Learn.
The differents task we have to are : 
 - 1 In a Notebook, create a custom loading class inspired from https://github.com/UKPLab/sentence-transformers/blob/6fcfdfb30f9dfcc5fb978c97ce02941a7aa6ba63/sentence_transformers/datasets/SentenceLabelDataset.py.

 - 2 Build a training pipeline and fine-tune a ```distilbert-base-nli-mean-tokens``` model with your custom loading class.

 - 3 Find an Approximates Approximate Nearest Neighbor library

 - 4 Build a basic prediction pipeline:
    - a. Vectorize the training set with your fine-tuned sBERT model
    - b. Index all these vectors with your ANN library
    - c. Build a barebone kNN classifier where new text input gets predicted the same label as that of the closest neighbor from the index
    - d. Benchmark this pipeline with the test set
    - e. Compare the results with the pretrained sBERT model

 - 5 Create a simple Python REST API that serves this prediction via a ```/predict``` route.


 We organize this repository like this : 

 - ```notebook``` folder: Get the notebook for question 1 and 2
 - ```api``` folder : for the REST API
 - ```requirements.txt``` file for dependances
 - ```models``` folder : to saved models
 - ```bin``` folder : fine tuning code for one shot classfication 



 ## HOW TO USE THIS REPOSITORY

  - Clone this repository with following command ```git clone https://github.com/amaelDago/NewsgroupsClassifier.git```
  - Move to repository : ```cd NewsgroupsClassifier```
  - Build docker image : ```docker build -t <myimage> .``` . 
  
    Notice : When you run this script, models and train_embedding have automocally loaded

  - Run container : ```docker run -d --name <mycontainer> -p 8000:80 <myimage>``` (-d for detach mode)
  - Go to your host at port 8000 to see the API (```http://localhost:8000 ```) 
  - For inference you can use : 
      - Fast API swagger to test at :  ```http://localhost:8000/docs```
      - HTTP request via curl : ```curl -X 'POST' 'http://localhost:8001/predict/<sentence>?sentence=mysentence' -H 'accept: application/json' \-d ''```
      - Insert your sentence on your web browser like this : ```http://localhost:8001/predict/<sentence>?sentence=mysentence```

   Result looks like : 
      ```python
         {
      "id": 17,
      "label": "talk.politics.mideast",
      "distance": 155.3105926513672
      }
      ```

You can see colab notebook at this [link](https://colab.research.google.com/drive/136K1hUT0kDTq8QiOiweYcU8RcwUBBXYx?usp=sharing)

   