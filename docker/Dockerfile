FROM python:3.10.14-bookworm

USER root

# set environment variable
ENV WS=/workspace \
    DEBIAN_FRONTEND=noninteractive \
    TIMEZONE=Asia/Taipei

WORKDIR $WS

# Setting TimeZone
RUN apt -yq update && \
    apt -yq install tzdata && \
    ln -fs /usr/share/zoneinfo/${TIMEZONE} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Install package
COPY [ "./requirements.txt", "/tmp/requirements.txt" ]
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Change file permission
COPY ["entrypoint", "/usr/bin"]
RUN chmod u+x /usr/bin/entrypoint

# Setting entrypoint
ENTRYPOINT ["entrypoint"]