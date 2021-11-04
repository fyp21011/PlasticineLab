FROM liuyunhao1578/deformsim:latest

COPY . /opt/PlasticineLab/
RUN pip3 install -e /opt/PlasticineLab/
RUN mkdir -p ~/output

CMD /bin/bash
