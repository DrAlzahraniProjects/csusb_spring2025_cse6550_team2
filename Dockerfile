# syntax=docker/dockerfile:1

# Adapted from the Dockerfile overview page: https://docs.docker.com/build/concepts/dockerfile/

FROM python:3.9.21-slim-bookworm

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# install app
COPY app.py /

# final configuration
EXPOSE 2502
CMD ["python", "app.py"]