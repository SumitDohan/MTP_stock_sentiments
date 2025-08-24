#!/bin/bash

# Login to the registry
docker login 172.26.1.219 --username pensive_jemison --password 27Swetadey

# Pull the image from the registry
docker pull 172.26.1.219/pensive_jemison/finbert-mlops:latest

# Run the container (mapping port 8000 to 8000)
docker run -d --name finbert-app -p 8000:8000 172.26.1.219/pensive_jemison/finbert-mlops:latest
