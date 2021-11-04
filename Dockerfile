FROM liuyunhao1578/deformsim:latest

COPY . /opt/PlasticineLab/
RUN mkdir -p ~/output

CMD pip3 install -e /opt/PlasticineLab/ && /bin/bash
