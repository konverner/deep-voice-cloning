FROM python:3.9
MAINTAINER Konstantin Verner <konst.verner@gmail.com>
COPY . .
RUN pip install .