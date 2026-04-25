FROM nvcr.io/nvidia/deepstream-l4t:7.0-triton

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel $(nproc)

CMD ["/app/build/heimdall"]
