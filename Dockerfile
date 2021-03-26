FROM python:3.6-stretch

RUN apt-get update

RUN python -m pip install --upgrade pip

# deleting folder if already exist.
RUN rm -f add_on_recommender

# create new folder and make it working dir.
RUN mkdir /add_on_recommender
RUN cd /add_on_recommender
WORKDIR .

RUN mkdir predicted_data

# copy all content of the repo on docker.
COPY . .

# install requirements.
RUN pip3 install -r requirements.txt

#docker cp <containerId>:/file/path/within/container /host/path/target