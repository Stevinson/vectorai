# Vector AI

This repo is the interview project for Vector AI - contained are Dagster pipelines to 
train a CNN, stream data to Kafka, and fetch this data to make predictions and save in 
a database.

Dagster was chosen as the orchestration tool for ease of packaging and for 
observability into the state of jobs and data.

Docker Compose is used to set up the components of the project. Namely:

* Dagster to trigger tasks (separate containers for the FE, source code and Dagster daemon)
* MLFlow for logging the ML artefacts
* Kafka 
* Zookeeper  
* A PostgreSQL database for Dagster artefacts
* A PostgreSQL database for logging the predictions

There are 3 pipelines that can be triggered:

* Training a model:
  * Loads data from `/data/external/<directory_dataset_name>`
  * Splits the data into train/validation/test sets
  * Trains a simple CNN
  * Logs loss, accuracy and epoch info to MLFlow
  * Saves the model
  * Batch size, image size, epochs, data paths are all configurable
* Streaming data to Kafa or Google Pub/Sub:
  * Loads sample data from `/data/external/<directory_dataset_name>/stream_sample.p`
  * Sends this data to the selected service
  * On each run data can be either sent to Kafka or Pub/Sub, configurable in the stream_config.yaml
* Prediction:
  * Loads the model
  * Streams the data from Kafka and Pub/Sub
  * `stream_timeout_ms` defines the time that the job will block and wait for new messages 
    to arrive in the topics. 
  * Makes class predictions on the images
  * Saves the image, class prediction and timestamp to a database

NB. The following restrictions apply:

* The images must be greyscale (i.e. 1 value per pixel) and normalised
* The data to be streamed to Kafka has to be labelled data
* This has only been tested on Mac - though Docker should provide some level of interoperability
* The implementation has been tested on mnist and cifar10 datasets with a focus on the 
  infrastructure. No effort has been made to optimise the classification performance.
* Google Pub/Sub is hard-coded to a particular project to which you will need the credentials .json file 

## Setup Instructions

You must have Docker, docker-compose and git LFS installed. 

One piece of manual path setting needs to be done due to a seeming incompatibility with 
relative paths. In `dagster.yaml` the following lines need to be changed to represent
your filesystem.

```
# NB! These need to be manually set to your filesystem
- /Users/edward/github/vector_ai/src:/opt/dagster/app/src
- /Users/edward/github/vector_ai/models:/opt/dagster/app/models
- /Users/edward/github/vector_ai/data:/opt/dagster/app/data
```

Then the following command needs to be run.

```
docker-compose up
```

NB. If you shut down the containers you will need to run `docker-compose rm -svf` to 
ensure that Zookeeper does not cause an error before restarting the docker containers.

## Usage

* The Dagster UI will be available at `localhost:3000` (alternatively you could launch 
  jobs using the Python)
  * The Dagster UI will display 3 jobs on the left pane: `train_model`, `stream_model`, 
    and `predict_model`, which should be run in this order.
  * To run a job click on the job then click on the 'Launchpad' tab. This will load the 
    preconfigured settings. Note these can be changed here or in the raw yaml files at
    `src/configs` to change settings such as batch size, number of epochs, etc.
  * Click on 'Launch Run'  

* The MLFlow UI will be available at `localhost:5000`

## Debugging

* If you see a `ChildProcessCrashException` you will need to increase Docker's memory settings
* If git lfs does not download the data, it can be fetched by running `src/data_helpers.py`. This 
  will require the environment to be created with:
```
conda env create
conda activate vectorai
pip install poetry 
poetry config virtualenvs.create false --local
poetry install
```


Project Organization 
------------

    ├── build                    <- Dockerfiles and.env files
    │
    ├── data
    │
    │   ├── external             <- Data from third party sources.
    │   ├── processed            <- Final data sets for modeling
    │   └── tfrecords            <- Data saved as .tfrecord
    │
    ├── models                   <- Trained and serialized models
    │
    ├── src                      <- Python source code for use in this project
    │   │
    │   ├── configs              <- Dagster yaml files for job configs
    │   │
    │   ├── predict_model.py     <- Use streamed data to make predictions 
    │   │
    │   ├── stream_data.py       <- Stream data to Kafka
    │   │
    │   └── train_model.py       <- Load data and train model
    │
    ├── models                   <- Trained and serialized models
    │
    ├── tests                    <- Integration tests of Dagster jobs
    │
    └── docker-compose.yaml      <- Setup of containerised solution

--------
