# FROM ubuntu:latest
FROM python:3.8.1
# COPY ./*.py /exp/
COPY ./serve.py /exp/
COPY ./svm_gamma_0.0005_C_0.5.joblib /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install -U scikit-learn
RUN pip3 install -r /exp/requirements.txt
WORKDIR /exp
EXPOSE 5000
CMD ["python3", "./serve.py" ]
# docker run -p 5000:5000 -it exp:v1