## Add-On Recommender

It is a simple add-on products’ recommendation based on the Bayesian
probability that two items (of a specific product_type, brand, and category)
will be bought or viewed together.

### TechStack:
- Docker
- Python 3.6
- Pandas

### How it works:
- Data is read from input_data/ folder.
- Then it is transformed and put in the Recommender to get trained models.
- Predictor is run to get the required predictions.

### How to run:
To build the docker images:
> docker-compose up --build

To train model and add recommendation:
> docker-compose run recommender

### Copy predicted data to local system:
> docker cp recommender:/predicted_data/predict.json  predicted_data/

### Tests
To run tests:
> docker-compose run tests

---------------------------------------------------------------------------------------------------------
**Note:**
```
As data was quite large, I was unable to upload it on Github.
So I took chunk of records of all_products_data and viewed_together_data to upload on github.
Please replace the data in "input_data" directory with the original data files. Thanks.
```