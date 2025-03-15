# syntax=docker/dockerfile:1

# Adapted from the Dockerfile overview page: https://docs.docker.com/build/concepts/dockerfile/
# NOTE: FAISS, LangChain, and streamlit require Python 3.9-3.13
FROM python:3.10-slim

# Install dependencies for running Apache
RUN apt-get update \
	&& apt-get install -y apache2 apache2-utils libapache2-mod-proxy-uwsgi libxml2-dev libxslt-dev

WORKDIR /app

# Copy requirements.txt into image
COPY "requirements.txt" /app/

# Install pip and necessary libraries
RUN apt-get update \
	&& apt-get install -y python3 python3-pip \
	&& pip install -r "requirements.txt" \
	&& apt-get clean \
	&& echo "ProxyPass /team2s25 http://localhost:2502/team2s25" >> /etc/apache2/sites-available/000-default.conf \
	&& echo "ProxyPassReverse /team2s25 http://localhost:2502/team2s25" >> /etc/apache2/sites-available/000-default.conf \
	&& a2enmod proxy proxy_http rewrite

# Copy the entire scripts folder into /scripts
COPY data/index/ /app/data/index/

# Copy app.py into the container
COPY app.py /app/

# Copy documentation.ipynb into /docs
#COPY documentation.ipynb /docs/documentation.ipynb

# Expose ports for streamlit and jupyter
# TODO: Final app must accept both IPv4 and IPv6 traffic; currently it only accepts IPv4(?)
# TODO: Currently localhost URL works, but network and external URLs cannot connect
EXPOSE 2502/tcp 

# TODO: Are we allowed to use a config.toml file instead of specifying each flag individually?
ENTRYPOINT ["sh", "-c", "apache2ctl start & streamlit run app.py --browser.gatherUsageStats=false --server.baseUrlPath='/team2s25' --server.port=2502 --theme.backgroundColor=#0065BD --theme.primaryColor=#808284 --theme.secondaryBackgroundColor=#808284 --theme.textColor=#FFFFFF"]