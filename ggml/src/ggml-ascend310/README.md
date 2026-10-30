# Legacy ACL backend for Ascend 310

This experimental backend runs supported `llama.cpp` matrix multiplications on a first-generation Huawei Ascend 310 NPU through the legacy ACL operator API in CANN 20.0.RC1. It is a separate backend from `GGML_CANN`, which uses the standard CANN integration documented in [the CANN build guide](../../../docs/build.md#cann).

Use this backend only when the machine has **Ascend 310** and a compatible legacy CANN 20 installation. Ascend 310P, 910, 910B, and 910C are different targets; this backend does not claim support for them.

## What is supported

- The backend advertises one NPU device, `ASCEND3100` (device index 0).
- It implements eligible `GGML_OP_MUL_MAT` nodes. Other graph operations are not implemented by this backend and must run on a backend that supports them, usually the CPU.
- Weight tensors may be F16, F32, or a quantized GGML type with a registered conversion-to-float routine. Activations must be F16 or F32; the matrix-multiplication output is F32.
- The inner dimension `K` must be positive and divisible by 16. The weight matrix must be two-dimensional. The output-row dimension `M` must be 1 or divisible by 16. The activation/output columns `N` are padded to a multiple of 16 for the Ascend 310 operator.
- Matrix tensors must be contiguous, dimensions must match, and activation batch dimensions must match the output. Unsupported shapes are not accepted by this backend.
- Quantized weights are converted to FP16 for execution. Converted weight tensors are cached on the NPU up to a 4 GiB backend cache limit; this reduces repeated conversion and transfer but consumes NPU memory.

The first use of each matrix shape invokes the CANN `atc` compiler to generate an offline operator model. Models are cached under `$GGML_ASCEND310_CACHE`, or by default `$HOME/.cache/ggml-ascend310`. Keep this directory writable and leave room for generated operator files.

## Requirements

- Linux x86_64 with an operational Ascend 310 driver. Check the device with `npu-smi info` before building.
- CANN toolkit **20.0.RC1**, including the legacy ACL and ACL operator compiler headers/libraries (`acl/acl.h`, `acl/acl_op.h`, `libascendcl`, and `libacl_op_compiler`). The toolkit root must contain `fwkacllib/` and `atc/`.
- GCC/G++ 8 is the compiler used for the validated build on this legacy toolchain. The path below is an example for the Huawei AI1S image; replace it if CANN is installed elsewhere.
- The backend prepares CANN's ATC `PATH`, Python module paths, and operator-path variables when it initializes. It also creates a Python 3.7 shim under the backend cache when `/usr/bin/python3.7` or `/usr/local/bin/python3.7` is available. The process must still be able to load CANN and driver shared libraries; configure `LD_LIBRARY_PATH` before starting the program.

The tested toolkit root was:

```sh
/usr/local/Ascend/ascend-toolkit/20.0.RC1/x86_64-linux_gcc7.3.0
```

The `gcc7.3.0` part is the toolkit's installation directory name. The tested `llama.cpp` build used GCC 8.4.0.

## Build

From the repository root, set `CANN_INSTALL_DIR` to the toolkit root and build the legacy backend explicitly:

```sh
export CANN_INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/20.0.RC1/x86_64-linux_gcc7.3.0

cmake -S . -B build-ascend310 \
  -DGGML_ASCEND310=ON \
  -DGGML_CANN=OFF \
  -DCANN_INSTALL_DIR="$CANN_INSTALL_DIR" \
  -DCMAKE_C_COMPILER=gcc-8 \
  -DCMAKE_CXX_COMPILER=g++-8 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-ascend310 --target llama-cli llama-server -j2
```

`CANN_INSTALL_DIR` can also be provided as `ASCEND_TOOLKIT_HOME`. The build fails during CMake configuration if the legacy ACL headers are missing. Keep the build directory separate from a regular `GGML_CANN` build so the selected backend and toolkit settings are unambiguous.

## Runtime environment

The dynamic loader needs the CANN and driver library directories before `llama-cli` or `llama-server` starts:

```sh
export CANN_INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/20.0.RC1/x86_64-linux_gcc7.3.0
export LD_LIBRARY_PATH="$CANN_INSTALL_DIR/fwkacllib/lib64:$CANN_INSTALL_DIR/acllib/lib64:$CANN_INSTALL_DIR/toolkit/lib64:$CANN_INSTALL_DIR/atc/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
```

Check the NPU before launching:

```sh
npu-smi info
```

Select the backend with `-dev ASCEND3100` (CLI) or `--device ASCEND3100` (server), and request NPU layers with `-ngl` / `--gpu-layers`. A local CLI smoke test is:

```sh
./build-ascend310/bin/llama-cli \
  -m /path/to/model.gguf \
  -p "Reply with exactly the word ready." \
  -n 8 -c 4096 -ngl 99 -dev ASCEND3100 -nkvo
```

For a local OpenAI-compatible server:

```sh
./build-ascend310/bin/llama-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 4096 --batch-size 16 --ubatch-size 16 \
  --gpu-layers 99 --device ASCEND3100 --no-kv-offload
```

`--no-kv-offload` keeps the KV cache on the CPU. This can preserve NPU memory for weights and matrix work on an 8 GiB device, but it uses host RAM and does not make long-context prefill fast. Choose context and batch sizes based on measured RAM, NPU memory, and latency; the model's advertised maximum is not a guarantee that this backend can run that maximum efficiently.

Verify both the API response and NPU activity:

```sh
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","messages":[{"role":"user","content":"Reply with exactly ready."}],"max_tokens":8,"temperature":0}'
npu-smi info
```

An HTTP success response by itself does not prove NPU execution; confirm that `npu-smi` shows NPU memory or AICore activity while inference runs.

## Validation and known limitations

The [MiniCPM5-2B model card](https://huggingface.co/openbmb/MiniCPM5-2B) advertises a native context length of 131,072 tokens. That is a model limit, not a guarantee that this backend can run the full window efficiently. The initial validation used a Huawei AI1S VM with Ascend 310 (8 GiB), CANN 20.0.RC1, and the `MiniCPM5-2B-Q4_K_M.gguf` model. A retrieval prompt with 3,030 prompt tokens completed with HTTP 200 in 313.6 seconds and returned the code placed near the end of the prompt. `npu-smi` showed about 6,389 / 8,192 MiB in use and AICore activity during execution. This is a functional long-context check, not a performance target; the run was slow.

- This backend is experimental and currently supports only the eligible matrix-multiplication subset above; it is not a general replacement for the standard CANN backend.
- Tested CANN 20.0.RC1 can report AICore exceptions and a device health state of `Warning` after unsupported operator shapes are submitted. Confirm correctness with a known-answer request, inspect service logs, and monitor `npu-smi` rather than treating a successful build as sufficient validation.
- On the tested legacy runtime, a one-shot CLI process can print a correct answer and then abort during CANN/HDC teardown. The long-lived `llama-server` service remained available and answered HTTP requests. This teardown issue is not proof that every process exit is clean.
- The `test-backend-ops` executable did not link on the tested legacy build because its link command omitted pthread (`pthread_create ... DSO missing from command line`). The CLI and server targets did build; this document does not claim that the full backend test suite passes.
- Flash Attention is not supported by this CANN 20 / Ascend 310 combination and is disabled by the runtime.

The upstream CANN backend and its own model/device matrix are documented in [docs/backend/CANN.md](../../../docs/backend/CANN.md). Please report the exact NPU model, driver version, toolkit version, model format, failing operation shape, and relevant log excerpt when filing an issue about this legacy path.
