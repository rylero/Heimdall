# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    git \
    ccache \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only build config — layer cached until deps change
COPY CMakeLists.txt .
COPY src/models/bbox_parser/CMakeLists.txt src/models/bbox_parser/CMakeLists.txt
COPY proto/ proto/
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache

# Build third-party deps — cached until CMakeLists.txt changes
RUN --mount=type=cache,target=/root/.ccache \
    cmake --build build --parallel $(nproc) \
    --target libzmq libprotobuf protoc Catch2WithMain

# Build our code — deps already compiled, only our files recompile on source changes
COPY . .
RUN --mount=type=cache,target=/root/.ccache \
    cmake --build build --parallel $(nproc)

CMD ["/app/build/heimdall"]
