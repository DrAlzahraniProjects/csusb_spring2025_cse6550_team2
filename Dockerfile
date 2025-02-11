# syntax=docker/dockerfile:1

# Adapted from the Dockerfile overview page: https://docs.docker.com/build/concepts/dockerfile/
# NOTE: FAISS, LangChain, and streamlit require Python 3.9-3.13
FROM python:3.10-slim

# Copy requirements.txt into image
COPY "requirements.txt" "requirements.txt"

# Install pip and necessary libraries
RUN apt-get update \
	&& apt-get install -y python3 python3-pip \
	&& pip install -r "requirements.txt"

# install app
COPY app.py /
COPY documentation.ipynb /docs/documentation.ipynb

# final configuration
# TODO: Final app must accept both IPv4 and IPv6 traffic; currently it only accepts IPv4(?)
# TODO: Currently localhost URL works, but network and external URLs cannot connect
EXPOSE 2502/tcp 2512/tcp

# RUN jupyter notebook \
# 	...

RUN echo "#!/bin/bash\n\
jupyter notebook --ip=0.0.0.0 --port=2512 --no-browser --allow-root --log-level=CRITICAL --NotebookApp.base_url='team2s25/jupyter' --ServerApp.root_dir='/docs/' --ServerApp.token='' &\n\
streamlit run app.py --browser.gatherUsageStats=false --server.baseUrlPath='team2s25' --server.port=2502 --theme.backgroundColor=#0065BD --theme.primaryColor=#808284 --theme.secondaryBackgroundColor=#808284 --theme.textColor=#FFFFFF" > /start.sh && chmod +x ./start.sh

# TODO: Are we allowed to use a config.toml file instead of specifying each flag individually?
# TODO: Generalize browser.serverAddress
ENTRYPOINT ["./start.sh"]