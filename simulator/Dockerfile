FROM openroad/ubuntu20.04-dev

WORKDIR /app

RUN apt-get -y update && apt-get -y install \
    python3-pip \
    libprotobuf-dev  \
    protobuf-compiler \
    pkg-config \
    locales \
    curl

RUN locale-gen en_US.UTF-8

RUN git clone -b "v4.3.4" https://github.com/zeromq/libzmq.git /tmp/installer/libzmq && \
    cd /tmp/installer/libzmq && \
    cmake -B build . && \
    cmake --build build -j $(nproc) --target install

RUN git clone -b "v4.8.1" https://github.com/zeromq/cppzmq.git /tmp/installer/cppzmq && \
    cd /tmp/installer/cppzmq && \
    cmake -B build . && \
    cmake --build build -j $(nproc) --target install

RUN rm -rf /tmp/installer

RUN apt-get remove pkg-config -y && \
    apt-get clean -y && \
    apt-get autoclean -y && \
    apt-get autoremove -y

RUN pip install fastapi==0.97.0
RUN pip install uvicorn==0.22.0
RUN pip install pyzmq==25.1.0
RUN pip install orjson==3.9.1

RUN printf '#!/bin/bash\n[ -f /app/start_container ] && /app/start_container || /bin/bash' > /usr/bin/entrypoint
RUN chmod +x /usr/bin/entrypoint

CMD ["entrypoint"]
