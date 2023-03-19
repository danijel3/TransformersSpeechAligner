#!/usr/bin/env bash

docker run -it --rm -p 8080:80 --name web -v $PWD/website:/usr/share/nginx/html nginx