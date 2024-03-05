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
      libmkl-dev \
      libfftw3-dev \
      libgmp-dev \
      libhdf5-dev \
      libopenmpi-dev \
      gunicorn \
      python3-dev \
      python3-mako \
      python3-mpi4py \
      python3-pip \
      python3-skimage \
      python3-gunicorn \
      python3-pandas \
      libpython3-dev \
      && \
      apt-get autoremove --purge -y && \
      apt-get autoclean -y && \
      rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Install Python dependencies.
RUN pip3 install --no-cache-dir --upgrade Flask dash[compress] dash-daq dash-bootstrap-components dash-extensions plotly cython setuptools scipy

ARG CFLAGS="-fopenmp -m64 -march=x86-64 -mtune=generic -O3"
ARG CXXFLAGS="-fopenmp -m64 -march=x86-64 -mtune=generic -O3"
ARG LDFLAGS="-ldl -lm"
ARG FFLAGS="-fopenmp -m64 -march=x86-64 -mtune=generic -O3"
ENV MKL_THREADING_LAYER=SEQUENTIAL
ENV MKL_INTERFACE_LAYER=GNU,LP64
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

RUN git clone -b v1.26.4 --depth 1 https://github.com/numpy/numpy.git /src/numpy \
    && cd /src/numpy \
    && git submodule update --init \
    && NPY_BLAS_ORDER=MKL NPY_LAPACK_ORDER=MKL python3 setup.py install

# Create a working directory.
RUN mkdir /fermisee
WORKDIR /fermisee

# Copy the rest of the codebase into the image
COPY . ./

# Finally, run gunicorn.
# CMD [ "gunicorn", "--workers=4", "--threads=1", "-b 0.0.0.0:9375", "app:server"]
# or run in debug mode
CMD ["python3", "app.py"]
