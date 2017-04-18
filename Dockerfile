FROM acusensehub/scikit-opencv:py-2.7

VOLUME ["/home/_data", "/home/_inputs", "/home/_shared_outputs", "/home/src", "/home/_snapshots"]

# Setup environment variables
ENV INPUT_DIR=/home/_inputs
ENV SHARED_OUTPUT_DIR=/home/_shared_outputs
ENV SNAPSHOTS_DIR=/home/_snapshots
ENV DATA_DIR=/home/_data
ENV SRC_DIR=/home/src

# Run commands to make code work
RUN apt-get update -y

# Numpy / Scipy reqs
RUN apt-get install -y  ipython \
		        ipython-notebook \
                        python-pandas \
		        python-sympy

RUN mkdir -p /home/src

COPY src /home/src

RUN find /home/src/scripts -name "*.sh" -exec chmod +x {} +

# Working directory: this is where unix scripts will run from
WORKDIR /home/src
