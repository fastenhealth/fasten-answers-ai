#!/bin/bash

# Run Elasticsearch container with 512MB memory limit
docker run --rm -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.security.http.ssl.enabled=false" \
  --memory="512m" \
  -v data:/usr/share/elasticsearch/data \
  --name elasticsearch \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.1
