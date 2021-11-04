FROM liuyunhao1578/deformsim:0.1

COPY . /opt/PlasticineLab-latest/
RUN pip3 install -e /opt/PlasticineLab-latest/
RUN mkdir -p ~/output

CMD /bin/bash
