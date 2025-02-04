# syntax=docker/dockerfile:1

# Adapted from the Dockerfile overview page: https://docs.docker.com/build/concepts/dockerfile/
# NOTE: Streamlit requires Python 3.9-3.13
FROM python:3.9-slim

# Copy requirements.txt into image
COPY "requirements.txt" "requirements.txt"

# install app dependencies
RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && pip install -r "requirements.txt"

# install app
COPY app.py /

# final configuration
EXPOSE 2502
# TODO: Streamlit used port 8501 instead of 2502. There is likely a flag that needs to be specified.
CMD ["streamlit", "run", "app.py"]