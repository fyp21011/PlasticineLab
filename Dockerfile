FROM liuyunhao1578/deformsim:0.1

RUN mkdir -p /opt/PlasticineLab-latest/
COPY ./* /opt/PlasticineLab-latest/
RUN pip3 install -e /opt/
RUN mkdir -p ~/output

CMD /bin/bash
