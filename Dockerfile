# syntax=docker/dockerfile:1
# Adapted from the Dockerfile overview page: https://docs.docker.com/build/concepts/dockerfile/
# NOTE: FAISS, LangChain, and streamlit require Python 3.9-3.13
# NOTE: Alpine would be smaller but Streamlit doesn't seem to work with it
FROM python:3.13-slim AS builder

# Copy requirements.txt into image
COPY "requirements.txt" .
RUN python3 -m venv /env \
    && /env/bin/pip install --upgrade pip \
    && /env/bin/pip install -r "requirements.txt" --no-cache-dir -U --upgrade-strategy eager \
    && echo "    - Installed Python libraries."

FROM python:3.13-slim
COPY --from=builder /env /env
ENV PATH="/env/bin:$PATH"
WORKDIR /app

# Copy the app file, the scraper file, and the crontab file
COPY app.py /app/
COPY scraper.py /app/
COPY crontab.txt /app/

# Install dependencies for running Apache and libraries needed by scraper (like libxml2-dev for lxml), and install cron
# If 'crontab: not found' error persists, check the output of this RUN command carefully for apt-get errors.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc apache2 apache2-utils libapache2-mod-proxy-uwsgi libxml2-dev libxslt-dev cron \
    && apt-get upgrade -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Add the crontab file to the system cron jobs
RUN crontab /app/crontab.txt
# Create the log file directory and file
RUN mkdir -p /var/log && touch /var/log/scraper_cron.log


# Expose port for Streamlit
EXPOSE 2502/tcp

# Ensure 000-default.conf exists in the same directory as the Dockerfile
#COPY "000-default.conf" "/etc/apache2/sites-available/000-default.conf"
RUN echo "ProxyPass /team2s25 http://localhost:2502/team2s25" >> /etc/apache2/sites-available/000-default.conf \
    && echo "ProxyPassReverse /team2s25 http://localhost:2502/team2s25" >> /etc/apache2/sites-available/000-default.conf \
    && echo "RewriteEngine On" >> /etc/apache2/sites-available/000-default.conf \
    && echo "RewriteRule \"^/team2s25/(.*)\" \"ws://localhost:2502/team2s25/$1\" [P,L]" >> /etc/apache2/sites-available/000-default.conf \
    && a2enmod proxy proxy_http rewrite

# Ensure the data/index directory exists, even if empty initially
RUN mkdir -p /app/data/index

# TODO: Are we allowed to use a config.toml file instead of specifying each flag individually?
# Update ENTRYPOINT to start cron, then apache, then streamlit
# Use a simple script to keep processes running and start them
ENTRYPOINT ["/bin/sh", "-c", "cron -f & apache2ctl start & streamlit run app.py --server.baseUrlPath=/team2s25 --server.port=2502 --theme.backgroundColor=#0065BD --theme.primaryColor=#808284 --theme.secondaryBackgroundColor=#808284 --theme.textColor=#FFFFFF --browser.gatherUsageStats=false; wait"]