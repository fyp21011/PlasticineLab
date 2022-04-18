FROM liuyunhao1578/deformsim:latest

COPY . /opt/PlasticineLab/
RUN pip3 install -e /opt/PlasticineLab/

CMD /bin/bash
