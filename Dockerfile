FROM liuyunhao1578/deformsim:latest

COPY . /opt/PlasticineLab/
RUN apt update && apt autoclean && apt install subversion pulseaudio blender -y
RUN pip3 install -e /opt/PlasticineLab/

CMD /bin/bash
