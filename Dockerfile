FROM python:3.10-slim-bookworm
COPY . /app
WORKDIR /app 
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python app.py
EXPOSE 8000

