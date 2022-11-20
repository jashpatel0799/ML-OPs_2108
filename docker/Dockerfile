# FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
# COPY ./serve.py /exp/
COPY ./models/*.joblib /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install -U scikit-learn
RUN pip3 install -r /exp/requirements.txt
WORKDIR /exp
EXPOSE 5000
ENV CLF_MODEL="svm"
ENV R_S="42"
# CMD ["python", "./plot_final.py", "--clf_model", ${CLF_MODEL}, "--random_state", ${R_S}]
CMD python plot_final.py --clf_model ${CLF_MODEL} --random_state ${R_S}
# docker run -p 5000:5000 -it exp:v1