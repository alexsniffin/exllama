FROM exllama-web:latest as build

COPY --chown=$RUN_UID ./example_flask.py /app

WORKDIR /app

# Create application state directory and install python packages
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install -r requirements-web.txt

USER root

STOPSIGNAL SIGINT

EXPOSE 8080
ENTRYPOINT ["python3", "/app/example_flask.py"]
