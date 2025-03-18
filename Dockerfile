# syntax=docker/dockerfile:1

# Adapted from the Dockerfile overview page: https://docs.docker.com/build/concepts/dockerfile/
# NOTE: FAISS, LangChain, and streamlit require Python 3.9-3.13
# NOTE: Alpine would be smaller but Streamlit doesn't seem to work with it
FROM python:3.10-slim AS builder

# Copy requirements.txt into image
COPY "requirements.txt" .

RUN python3 -m venv /env \
	&& /env/bin/pip install --upgrade pip \
	&& /env/bin/pip install -r "requirements.txt" --no-cache-dir
# && echo "    - Installed Python libraries."

FROM python:3.10-slim

COPY --from=builder /env /env

ENV PATH="/env/bin:$PATH"

WORKDIR /app

# Copy the entire scripts folder into /scripts
COPY app.py data/index/ /app/

# Install dependencies for running Apache
RUN apt-get update \
	&& apt-get install -y apache2 apache2-utils libapache2-mod-proxy-uwsgi libxml2-dev libxslt-dev --no-install-recommends \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

COPY "000-default.conf" "/etc/apache2/sites-available/000-default.conf"

RUN a2enmod proxy proxy_http rewrite

# Expose port for Streamlit
# TODO: Final app must accept both IPv4 and IPv6 traffic; currently it only accepts IPv4(?)
# TODO: Currently localhost URL works, but network and external URLs cannot connect
EXPOSE 2502/tcp

# TODO: Are we allowed to use a config.toml file instead of specifying each flag individually?
ENTRYPOINT ["sh", "-c", "apache2ctl start & streamlit run app.py --server.baseUrlPath=/team2s25 --server.port=2502 --theme.backgroundColor=#0065BD --theme.primaryColor=#808284 --theme.secondaryBackgroundColor=#808284 --theme.textColor=#FFFFFF --browser.gatherUsageStats=false"]