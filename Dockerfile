# syntax=docker/dockerfile:1

# Base image with Python 3.13
# NOTE: FAISS, LangChain, and Streamlit require Python 3.9–3.13
FROM python:3.13-slim AS builder

# Copy dependency list into the image
COPY "requirements.txt" .

# Create virtual environment and install Python dependencies
RUN python3 -m venv /env \
	&& /env/bin/pip install --upgrade pip \
	&& /env/bin/pip install -r "requirements.txt" --no-cache-dir -U --upgrade-strategy eager

# Final image
FROM python:3.13-slim

# Copy virtual environment from builder
COPY --from=builder /env /env

# Set path to use the virtual environment
ENV PATH="/env/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py /app/
COPY crawler.py /app/             
COPY data /app/data              

# Install Apache and required modules
RUN apt-get update \
	&& apt-get install -y gcc apache2 apache2-utils libapache2-mod-proxy-uwsgi libxml2-dev \
	&& apt-get upgrade -y \
	&& apt-get clean -y \
	&& rm -rf /var/lib/apt/lists/*

# Expose the Streamlit port
EXPOSE 2502/tcp

# Add proxy and rewrite rules to Apache config
RUN echo "ProxyPass /team2s25 http://localhost:2502/team2s25" >> /etc/apache2/sites-available/000-default.conf \
	&& echo "ProxyPassReverse /team2s25 http://localhost:2502/team2s25" >> /etc/apache2/sites-available/000-default.conf \
	&& echo "RewriteEngine On" >> /etc/apache2/sites-available/000-default.conf \
	&& echo "RewriteRule /team2s25/(.*) ws://localhost:2502/team2s25/\$1 [P,L]" >> /etc/apache2/sites-available/000-default.conf \
	&& a2enmod proxy proxy_http rewrite

# Start the container by:
# 1. Running crawler.py once
# 2. Starting Apache in background
# 3. Launching the Streamlit app
ENTRYPOINT ["sh", "-c", "python /app/crawler.py && apache2ctl start & streamlit run app.py --server.baseUrlPath=/team2s25 --server.port=2502 --theme.backgroundColor=#0065BD --theme.primaryColor=#808284 --theme.secondaryBackgroundColor=#808284 --theme.textColor=#FFFFFF --browser.gatherUsageStats=false"]
