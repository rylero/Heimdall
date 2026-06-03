"""
Generate TensorRT INT8 calibration cache from preprocessed .npy tensors.

Run on the Jetson BEFORE run_trtexec_int8.sh.
Requires: tensorrt, pycuda (both ship with JetPack).

Usage (from training/calibration/ on Jetson):
  python gen_calib_cache.py \
    --onnx ../models/rfdetr_nano_480.onnx \
    --list calib_list.txt \
    --cache calib.cache \
    --imgsz 480
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="../models/rfdetr_nano_480.onnx")
    parser.add_argument("--list", default="calib_list.txt", dest="calib_list")
    parser.add_argument("--cache", default="calib.cache")
    parser.add_argument("--imgsz", type=int, default=480)
    args = parser.parse_args()

    try:
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import numpy as np
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Run this script on the Jetson — TensorRT and pycuda are included with JetPack.")
        sys.exit(1)

    list_file = Path(args.calib_list)
    if not list_file.exists():
        print(f"ERROR: {list_file} not found. Copy calibration/ from Windows training machine.")
        sys.exit(1)
    image_paths = [p.strip() for p in list_file.read_text().splitlines() if p.strip()]
    print(f"Calibration images: {len(image_paths)}")

    class Int8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, image_paths, cache_file, imgsz):
            super().__init__()
            self.image_paths = image_paths
            self.cache_file = cache_file
            self.imgsz = imgsz
            self.idx = 0
            nbytes = 3 * imgsz * imgsz * 4  # float32 bytes
            self.device_buf = cuda.mem_alloc(nbytes)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.idx >= len(self.image_paths):
                return None
            tensor = np.load(self.image_paths[self.idx]).astype(np.float32)
            # TRT expects (1, C, H, W) contiguous
            tensor = np.ascontiguousarray(tensor[np.newaxis])
            cuda.memcpy_htod(self.device_buf, tensor)
            self.idx += 1
            if self.idx % 10 == 0 or self.idx == len(self.image_paths):
                print(f"  {self.idx}/{len(self.image_paths)} images calibrated...")
            return [self.device_buf]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                print(f"Loading existing cache: {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                f.write(cache)
            print(f"Calibration cache written: {self.cache_file}")

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found. Copy rfdetr_nano_480.onnx from Windows machine.")
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    calibrator = Int8Calibrator(image_paths, args.cache, args.imgsz)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not onnx_parser.parse(f.read()):
            for i in range(onnx_parser.num_errors):
                print(onnx_parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    print("Building calibration engine (may take several minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("ERROR: TRT engine build failed.")
        sys.exit(1)

    print(f"\nCalibration cache ready: {args.cache}")
    print("Next: run ./run_trtexec_int8.sh")


if __name__ == "__main__":
    main()
