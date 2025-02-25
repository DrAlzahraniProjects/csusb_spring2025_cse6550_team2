# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt into image
COPY requirements.txt .

# Install pip and necessary libraries
RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && pip install -r requirements.txt \
	&& pip install langchain-huggingface \
	&& pip install langchain-community

# Copy the entire scripts folder into /app/scripts
COPY scripts/ ./scripts/

# Copy app.py into the container
COPY app.py .
COPY config.py .

# Copy documentation.ipynb into /docs
COPY documentation.ipynb /docs/documentation.ipynb

# Expose ports for streamlit and jupyter
EXPOSE 2502/tcp 2512/tcp

# Create a start script to run Jupyter Notebook and Streamlit
RUN echo "#!/bin/bash\n\
jupyter notebook --ip=0.0.0.0 --port=2512 --no-browser --allow-root --log-level=CRITICAL --NotebookApp.base_url='team2s25/jupyter' --ServerApp.root_dir='/docs/' --ServerApp.token='' &\n\
streamlit run app.py --browser.gatherUsageStats=false --server.baseUrlPath='team2s25' --server.port=2502 --theme.backgroundColor=#0065BD --theme.primaryColor=#808284 --theme.secondaryBackgroundColor=#808284 --theme.textColor=#FFFFFF" > /start.sh && chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
