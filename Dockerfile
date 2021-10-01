FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y software-properties-common apt-transport-https curl && \
    curl -L https://users.flatironinstitute.org/~ccq/triqs3/jammy/public.gpg | apt-key add - && \
    add-apt-repository "deb https://users.flatironinstitute.org/~ccq/triqs3/jammy/ /" -y && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      triqs \
      g++ \
      git \
      make \
      cmake \
      sudo \
      hdf5-tools \
      libboost-dev \
      intel-mkl \
      libfftw3-dev \
      libgmp-dev \
      libhdf5-dev \
      libopenmpi-dev \
      gunicorn \
      python3-dev \
      python3-mako \
      python3-scipy \
      python3-mpi4py \
      python3-pip \
      python3-plotly \ 
      python3-skimage \
      python3-gunicorn \
      python3-pandas \
      python3-flask \ 
      libpython3-dev \
      && \
      apt-get autoremove --purge -y && \
      apt-get autoclean -y && \
      rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Install Python dependencies.
RUN pip3 install dash[compress] dash-daq dash-bootstrap-components dash-extensions

ARG CFLAGS="-fopenmp -m64 -march=x86-64 -mtune=generic -O3 -Wl,--no-as-needed"
ARG CXXFLAGS="-fopenmp -m64 -march=x86-64 -mtune=generic -O3 -Wl,--no-as-needed"
ARG LDFLAGS="-ldl -lm"
ARG FFLAGS="-fopenmp -m64 -march=x86-64 -mtune=generic -O3"
ENV MKL_THREADING_LAYER=GNU
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

RUN echo "\n[mkl]" >> ~/.numpy-site.cfg && \
    echo "mkl_libs = mkl_def, mkl_gf_lp64, mkl_core, mkl_sequential" >> ~/.numpy-site.cfg && \
    echo "lapack_libs = mkl_def, mkl_gf_lp64, mkl_core, mkl_sequential" >> ~/.numpy-site.cfg && \
    pip3 install numpy --no-binary numpy --force-reinstall

# Create a working directory.
RUN mkdir /fermisee
WORKDIR /fermisee

# Copy the rest of the codebase into the image
COPY . ./

# Finally, run gunicorn.
CMD gunicorn --workers=8 --threads=1 -b 0.0.0.0:$PORT app:server
# or run in debug mode
# CMD ["python3", "app.py"]
