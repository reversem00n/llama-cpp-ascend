#include "ggml-ascend310.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <acl/acl.h>
#include <acl/acl_op.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>

#ifndef GGML_ASCEND310_CANN_ROOT
#define GGML_ASCEND310_CANN_ROOT "/usr/local/Ascend/ascend-toolkit/20.0.RC1/x86_64-linux_gcc7.3.0"
#endif

namespace {

struct ascend310_buffer_context {
    void * base;
};

struct ascend310_backend_context {
    int device;
    aclrtStream stream;
    std::unordered_map<const ggml_tensor *, void *> cached_weights;
    size_t cached_weight_bytes;
};

struct ascend310_device_context {
    int device;
    std::string name;
    std::string description;
};

struct ascend310_registry_context {
    std::vector<ggml_backend_dev_t> devices;
};

static ggml_backend_reg ggml_backend_ascend310_reg_struct;
static ggml_backend_buffer_type ggml_backend_ascend310_buffer_type;
static std::mutex ascend310_mutex;
static bool offline_model_dir_set = false;
static bool atc_environment_prepared = false;
static const int64_t ascend310_max_m_tile = 4096;
static const size_t ascend310_max_cached_weight_bytes = 4ULL * 1024 * 1024 * 1024;

// Ascend 310 MatMul supports N=1 (verified on the target CANN runtime). Keep
// decode at its true width; pad larger micro-batches to the kernel-friendly 16.
static int64_t matmul_n_extent(int64_t n) {
    return n == 1 ? 1 : ((n + 15) / 16) * 16;
}

static bool acl_ok(aclError status, const char * operation) {
    if (status == ACL_ERROR_NONE) {
        return true;
    }
    GGML_LOG_ERROR("ASCEND310: %s failed with ACL status %d\n", operation, (int) status);
    return false;
}

static void finalize_acl_runtime() {
    acl_ok(aclrtResetDevice(0), "reset device");
    acl_ok(aclFinalize(), "finalize ACL");
}

static bool make_dirs(const std::string & path) {
    std::string current;
    for (char ch : path) {
        current.push_back(ch);
        if (ch == '/' && current.size() > 1) {
            if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

static std::string cache_root() {
    const char * env_cache = getenv("GGML_ASCEND310_CACHE");
    if (env_cache && *env_cache) return env_cache;
    const char * home = getenv("HOME");
    return std::string(home && *home ? home : "/tmp") + "/.cache/ggml-ascend310";
}

static bool prepare_atc_environment(const std::string & cache) {
    if (atc_environment_prepared) {
        return true;
    }
    const std::string shim = cache + "/bin";
    if (!make_dirs(shim)) {
        return false;
    }
    const char * python_candidates[] = {"/usr/bin/python3.7", "/usr/local/bin/python3.7"};
    const char * python = nullptr;
    for (const char * candidate : python_candidates) {
        if (access(candidate, X_OK) == 0) {
            python = candidate;
            break;
        }
    }
    if (python != nullptr) {
        const char * names[] = {"python", "python3"};
        for (const char * name : names) {
            const std::string link = shim + "/" + name;
            unlink(link.c_str());
            if (symlink(python, link.c_str()) != 0) {
                return false;
            }
        }
    }

    const std::string root = GGML_ASCEND310_CANN_ROOT;
    const std::string te = root + "/atc/python/site-packages/te";
    const std::string py_path = te + ":" + te + "/te:" + root + "/fwkacllib/python/site-packages/te:" +
        root + "/fwkacllib/python/site-packages/te/te:" + root + "/atc/python/site-packages/topi:" +
        root + "/fwkacllib/python/site-packages/topi:" + root + "/atc/python/site-packages:" +
        root + "/fwkacllib/python/site-packages";
    const char * old_path = getenv("PATH");
    const std::string path = shim + ":" + root + "/atc/bin:" + root + "/fwkacllib/ccec_compiler/bin:" +
        root + "/atc/ccec_compiler/bin:" + (old_path ? old_path : "");
    const char * old_ld = getenv("LD_LIBRARY_PATH");
    const std::string ld = root + "/fwkacllib/lib64:" + root + "/acllib/lib64:" + root + "/toolkit/lib64:" +
        root + "/atc/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:" +
        (old_ld ? old_ld : "");
    const bool ok = setenv("PATH", path.c_str(), 1) == 0 && setenv("PYTHONPATH", py_path.c_str(), 1) == 0 &&
        setenv("LD_LIBRARY_PATH", ld.c_str(), 1) == 0 && setenv("ASCEND_OPP_PATH", (root + "/opp").c_str(), 1) == 0 &&
        setenv("OPP_PATH", (root + "/opp").c_str(), 1) == 0;
    atc_environment_prepared = ok;
    return ok;
}

static std::string model_dir_for(int64_t m, int64_t k, int64_t n) {
    char suffix[96];
    std::snprintf(suffix, sizeof(suffix), "/matmul_m%lld_k%lld_n%lld", (long long) m, (long long) k, (long long) n);
    return cache_root() + suffix;
}

static std::string model_file_for(int64_t m, int64_t k, int64_t n) {
    char name[192];
    std::snprintf(name, sizeof(name), "/models/0_MatMul_1_2_%lld_%lld_1_2_%lld_%lld_1_2_%lld_%lld.om",
        (long long) m, (long long) k, (long long) n, (long long) k, (long long) m, (long long) n);
    return cache_root() + name;
}

static bool compile_matmul(int64_t m, int64_t k, int64_t n, std::string & model_dir) {
    model_dir = model_dir_for(m, k, n);
    const std::string models = cache_root() + "/models";
    const std::string model_file = model_file_for(m, k, n);
    if (!prepare_atc_environment(cache_root())) {
        GGML_LOG_ERROR("ASCEND310: could not prepare CANN environment\n");
        return false;
    }
    if (access(model_file.c_str(), R_OK) == 0) {
        return true;
    }
    if (!make_dirs(model_dir) || !make_dirs(models)) {
        GGML_LOG_ERROR("ASCEND310: could not prepare ATC cache directories\n");
        return false;
    }
    const std::string json = model_dir + "/matmul.json";
    std::ofstream spec(json);
    if (!spec) {
        GGML_LOG_ERROR("ASCEND310: cannot write ATC operator spec %s\n", json.c_str());
        return false;
    }
    spec << "[{\"op\":\"MatMul\",\"input_desc\":["
         << "{\"format\":\"ND\",\"shape\":[" << m << "," << k << "],\"type\":\"float16\"},"
         << "{\"format\":\"ND\",\"shape\":[" << n << "," << k << "],\"type\":\"float16\"}],"
         << "\"output_desc\":[{\"format\":\"ND\",\"shape\":[" << m << "," << n << "],\"type\":\"float16\"}],"
         << "\"attr\":[{\"name\":\"transpose_x1\",\"type\":\"bool\",\"value\":false},"
         << "{\"name\":\"transpose_x2\",\"type\":\"bool\",\"value\":true}]}]";
    spec.close();

    const pid_t child = fork();
    if (child < 0) {
        GGML_LOG_ERROR("ASCEND310: fork for ATC failed: %s\n", strerror(errno));
        return false;
    }
    if (child == 0) {
        const std::string atc = std::string(GGML_ASCEND310_CANN_ROOT) + "/atc/bin/atc";
        const std::string singleop = "--singleop=" + json;
        const std::string output = "--output=" + models;
        char * const argv[] = {const_cast<char *>(atc.c_str()), const_cast<char *>(singleop.c_str()),
            const_cast<char *>("--soc_version=Ascend310"), const_cast<char *>(output.c_str()), nullptr};
        execv(atc.c_str(), argv);
        _exit(127);
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) {
            GGML_LOG_ERROR("ASCEND310: wait for ATC failed: %s\n", strerror(errno));
            return false;
        }
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || access(model_file.c_str(), R_OK) != 0) {
        GGML_LOG_ERROR("ASCEND310: ATC failed for MatMul M=%lld K=%lld N=%lld (status=%d)\n",
            (long long) m, (long long) k, (long long) n, status);
        return false;
    }
    return true;
}

struct acl_allocations {
    std::vector<void *> values;
    ~acl_allocations() {
        for (void * ptr : values) {
            aclrtFree(ptr);
        }
    }
    bool alloc(size_t bytes, void ** out) {
        if (!acl_ok(aclrtMalloc(out, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")) {
            return false;
        }
        values.push_back(*out);
        return true;
    }
};

static bool tensor_to_fp16(const ggml_tensor * tensor, std::vector<uint16_t> & result) {
    const int64_t k = tensor->ne[0];
    const int64_t rows = ggml_nelements(tensor) / k;
    result.resize((size_t) ggml_nelements(tensor));
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16) {
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t col = 0; col < k; ++col) {
                float value;
                const uint8_t * p = raw.data() + row * tensor->nb[1] + col * tensor->nb[0];
                if (tensor->type == GGML_TYPE_F32) {
                    std::memcpy(&value, p, sizeof(value));
                } else {
                    ggml_fp16_t half;
                    std::memcpy(&half, p, sizeof(half));
                    value = ggml_fp16_to_fp32(half);
                }
                result[(size_t) row * k + col] = ggml_fp32_to_fp16(value);
            }
        }
        return true;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        return false;
    }
    std::vector<float> row_data((size_t) k);
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(raw.data() + row * tensor->nb[1], row_data.data(), k);
        for (int64_t col = 0; col < k; ++col) {
            result[(size_t) row * k + col] = ggml_fp32_to_fp16(row_data[(size_t) col]);
        }
    }
    return true;
}

static bool cache_weight(ascend310_backend_context * ctx, const ggml_tensor * weights, const uint16_t ** data) {
    auto cached = ctx->cached_weights.find(weights);
    if (cached != ctx->cached_weights.end()) {
        *data = static_cast<const uint16_t *>(cached->second);
        return true;
    }

    const size_t bytes = (size_t) ggml_nelements(weights) * sizeof(uint16_t);
    if (bytes > ascend310_max_cached_weight_bytes - ctx->cached_weight_bytes) {
        return false;
    }

    std::vector<uint16_t> converted;
    if (!tensor_to_fp16(weights, converted)) {
        return false;
    }
    void * device_data = nullptr;
    if (!acl_ok(aclrtMalloc(&device_data, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate cached weights")) {
        return false;
    }
    if (!acl_ok(aclrtMemcpy(device_data, bytes, converted.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy cached weights")) {
        aclrtFree(device_data);
        return false;
    }
    ctx->cached_weights.emplace(weights, device_data);
    ctx->cached_weight_bytes += bytes;
    *data = static_cast<const uint16_t *>(device_data);
    return true;
}

static bool compute_matmul(ascend310_backend_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * input = dst->src[1];
    const int64_t k = weights->ne[0];
    const int64_t m = weights->ne[1];
    const int64_t n = input->ne[1];
    const int64_t n_exec = matmul_n_extent(n);
    const int64_t batches = input->ne[2] * input->ne[3];
    std::vector<uint16_t> a;
    std::vector<uint16_t> b;
    const uint16_t * cached_a = nullptr;
    const bool has_cached_a = cache_weight(ctx, weights, &cached_a);
    if ((!has_cached_a && !tensor_to_fp16(weights, a)) || !tensor_to_fp16(input, b)) {
        GGML_LOG_ERROR("ASCEND310: unsupported MUL_MAT input type (%s, %s)\n", ggml_type_name(weights->type), ggml_type_name(input->type));
        return false;
    }
    std::vector<float> output((size_t) ggml_nelements(dst));

    for (int64_t batch = 0; batch < batches; ++batch) {
        const size_t b_bytes = (size_t) n_exec * k * sizeof(uint16_t);
        std::vector<uint16_t> padded_b((size_t) n_exec * k, 0);
        std::copy_n(b.data() + (size_t) batch * n * k, (size_t) n * k, padded_b.data());
        for (int64_t m_start = 0; m_start < m; m_start += ascend310_max_m_tile) {
            const int64_t tile_m = std::min(ascend310_max_m_tile, m - m_start);
            std::string model_dir;
            if (!compile_matmul(tile_m, k, n_exec, model_dir)) {
                GGML_LOG_ERROR("ASCEND310: no offline MatMul model for M=%lld K=%lld N=%lld\n",
                    (long long) tile_m, (long long) k, (long long) n_exec);
                return false;
            }
            const size_t a_bytes = (size_t) tile_m * k * sizeof(uint16_t);
            const size_t c_bytes = (size_t) tile_m * n_exec * sizeof(uint16_t);
            acl_allocations buffers;
            void * da = has_cached_a ? const_cast<uint16_t *>(cached_a + (size_t) m_start * k) : nullptr;
            void * db = nullptr;
            void * dc = nullptr;
            if ((!has_cached_a && !buffers.alloc(a_bytes, &da)) || !buffers.alloc(b_bytes, &db) || !buffers.alloc(c_bytes, &dc)) {
                return false;
            }
            if ((!has_cached_a && !acl_ok(aclrtMemcpy(da, a_bytes, a.data() + (size_t) m_start * k, a_bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy weights")) ||
                !acl_ok(aclrtMemcpy(db, b_bytes, padded_b.data(), b_bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy activations")) {
                return false;
            }

            const int64_t a_dims[] = {tile_m, k};
            const int64_t b_dims[] = {n_exec, k};
            const int64_t c_dims[] = {tile_m, n_exec};
            aclTensorDesc * a_desc = aclCreateTensorDesc(ACL_FLOAT16, 2, a_dims, ACL_FORMAT_ND);
            aclTensorDesc * b_desc = aclCreateTensorDesc(ACL_FLOAT16, 2, b_dims, ACL_FORMAT_ND);
            aclTensorDesc * c_desc = aclCreateTensorDesc(ACL_FLOAT16, 2, c_dims, ACL_FORMAT_ND);
            aclDataBuffer * a_buf = aclCreateDataBuffer(da, a_bytes);
            aclDataBuffer * b_buf = aclCreateDataBuffer(db, b_bytes);
            aclDataBuffer * c_buf = aclCreateDataBuffer(dc, c_bytes);
            aclopAttr * attrs = aclopCreateAttr();
            bool ok = a_desc && b_desc && c_desc && a_buf && b_buf && c_buf && attrs;
            if (ok) {
                ok = acl_ok(aclopSetAttrBool(attrs, "transpose_x1", false), "set transpose_x1") &&
                     acl_ok(aclopSetAttrBool(attrs, "transpose_x2", true), "set transpose_x2");
            }
            if (ok) {
                const aclTensorDesc * input_desc[] = {a_desc, b_desc};
                const aclDataBuffer * input_buf[] = {a_buf, b_buf};
                const aclTensorDesc * output_desc[] = {c_desc};
                aclDataBuffer * output_buf[] = {c_buf};
                ok = acl_ok(aclopExecute("MatMul", 2, input_desc, input_buf, 1, output_desc, output_buf, attrs, ctx->stream), "aclopExecute(MatMul)") &&
                     acl_ok(aclrtSynchronizeStream(ctx->stream), "synchronize MatMul");
                if (!ok) {
                    GGML_LOG_ERROR("ASCEND310: MatMul model M=%lld K=%lld N=%lld (model N=%lld, tile offset=%lld), input types=%s/%s; input ne=[%lld,%lld,%lld,%lld], batches=%lld; descriptor bytes=%zu/%zu/%zu, buffer bytes=%zu/%zu/%zu\n",
                        (long long) tile_m, (long long) k, (long long) n, (long long) n_exec, (long long) m_start,
                        ggml_type_name(weights->type), ggml_type_name(input->type),
                        (long long) input->ne[0], (long long) input->ne[1],
                        (long long) input->ne[2], (long long) input->ne[3], (long long) batches,
                        aclGetTensorDescSize(a_desc), aclGetTensorDescSize(b_desc), aclGetTensorDescSize(c_desc),
                        a_bytes, b_bytes, c_bytes);
                }
            }
            std::vector<uint16_t> half_out((size_t) tile_m * n_exec);
            if (ok) {
                ok = acl_ok(aclrtMemcpy(half_out.data(), c_bytes, dc, c_bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy MatMul output");
            }
            if (attrs) aclopDestroyAttr(attrs);
            if (a_buf) aclDestroyDataBuffer(a_buf);
            if (b_buf) aclDestroyDataBuffer(b_buf);
            if (c_buf) aclDestroyDataBuffer(c_buf);
            if (a_desc) aclDestroyTensorDesc(a_desc);
            if (b_desc) aclDestroyTensorDesc(b_desc);
            if (c_desc) aclDestroyTensorDesc(c_desc);
            if (!ok) {
                return false;
            }
            for (int64_t row = 0; row < tile_m; ++row) {
                for (int64_t col = 0; col < n; ++col) {
                    output[(size_t) batch * m * n + (size_t) col * m + m_start + row] =
                        ggml_fp16_to_fp32(half_out[(size_t) row * n_exec + col]);
                }
            }
        }
    }
    ggml_backend_tensor_set(dst, output.data(), 0, output.size() * sizeof(float));
    return true;
}

static bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    // Model weights are allocated in the device buffer before graph execution.
    if (op->op == GGML_OP_NONE) {
        return true;
    }
    if (op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr || op->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    const int64_t k = a->ne[0];
    const int64_t m = a->ne[1];
    const int64_t n = b->ne[1];
    const ggml_type_traits * traits = ggml_get_type_traits(a->type);
    if (!traits || !traits->to_float || k != b->ne[0] || a->ne[2] != 1 || a->ne[3] != 1 ||
        b->ne[2] != op->ne[2] || b->ne[3] != op->ne[3] || m != op->ne[0] || n != op->ne[1]) {
        return false;
    }
    if (!(a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_F32 || ggml_is_quantized(a->type)) ||
        !(b->type == GGML_TYPE_F16 || b->type == GGML_TYPE_F32)) {
        return false;
    }
    return k > 0 && n > 0 && (k % 16 == 0) && (m == 1 || m % 16 == 0) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(op);
}

static enum ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * ctx = static_cast<ascend310_backend_context *>(backend->context);
    if (!acl_ok(aclrtSetDevice(ctx->device), "select device")) {
        return GGML_STATUS_FAILED;
    }
    std::lock_guard<std::mutex> lock(ascend310_mutex);

    // CANN 20.0 loads its offline operator-model index when aclopSetModelDir is
    // first called. Compile every dispatched graph shape, including exact N=1
    // decode, before installing the model directory; ATC files added afterward
    // are not visible to this process and aclopSetModelDir cannot be repeated.
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT || ggml_is_empty(node) ||
            (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            !supports_op(backend->device, node)) {
            continue;
        }
        const ggml_tensor * weights = node->src[0];
        const int64_t m = weights->ne[1];
        const int64_t k = weights->ne[0];
        const int64_t n = node->src[1]->ne[1];
        const int64_t n_exec = matmul_n_extent(n);
        for (int64_t m_start = 0; m_start < m; m_start += ascend310_max_m_tile) {
            const int64_t tile_m = std::min(ascend310_max_m_tile, m - m_start);
            std::string model_dir;
            if (!compile_matmul(tile_m, k, n_exec, model_dir)) {
                GGML_LOG_ERROR("ASCEND310: failed to prepare MatMul model M=%lld K=%lld N=%lld\n",
                    (long long) tile_m, (long long) k, (long long) n_exec);
                return GGML_STATUS_FAILED;
            }
        }
    }
    if (!offline_model_dir_set) {
        const std::string models = cache_root() + "/models";
        if (!acl_ok(aclopSetModelDir(models.c_str()), "set offline model dir")) {
            return GGML_STATUS_FAILED;
        }
        offline_model_dir_set = true;
    }

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_NONE ||
            node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE ||
            (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (!supports_op(backend->device, node) || !compute_matmul(ctx, node)) {
            GGML_LOG_ERROR("ASCEND310: failed to compute %s (%s)\n", node->name, ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const char * backend_name(ggml_backend_t) { return "ASCEND310"; }
static void backend_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ascend310_backend_context *>(backend->context);
    aclrtSetDevice(ctx->device);
    if (ctx->stream) aclrtSynchronizeStream(ctx->stream);
    for (const auto & cached : ctx->cached_weights) aclrtFree(cached.second);
    if (!ctx->cached_weights.empty()) {
        GGML_LOG_INFO("ASCEND310: released %zu cached weight tensors (%zu MiB)\n",
            ctx->cached_weights.size(), ctx->cached_weight_bytes / (1024 * 1024));
    }
    if (ctx->stream) aclrtDestroyStream(ctx->stream);
    delete ctx;
    delete backend;
}

static ggml_backend_t init_backend(ggml_backend_dev_t dev, const char *) {
    auto * dev_ctx = static_cast<ascend310_device_context *>(dev->context);
    if (!acl_ok(aclrtSetDevice(dev_ctx->device), "select device")) return nullptr;
    auto * ctx = new ascend310_backend_context{dev_ctx->device, nullptr, {}, 0};
    if (!acl_ok(aclrtCreateStream(&ctx->stream), "create stream")) {
        delete ctx;
        return nullptr;
    }
    static ggml_guid guid = {0x31,0x00,0xA5,0xCE,0x91,0x10,0x44,0x72,0x81,0x30,0xD2,0x11,0xA0,0x0C,0x10,0x01};
    return new ggml_backend{&guid, {backend_name, backend_free, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, graph_compute, nullptr, nullptr, nullptr}, dev, ctx};
}

static void buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ascend310_buffer_context *>(buffer->context);
    aclrtSetDevice(0);
    if (ctx->base) aclrtFree(ctx->base);
    delete ctx;
}
static void * buffer_base(ggml_backend_buffer_t buffer) { return static_cast<ascend310_buffer_context *>(buffer->context)->base; }
static void buffer_memset(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    aclrtSetDevice(0);
    aclrtMemset(static_cast<uint8_t *>(tensor->data) + offset, size, value, size);
}
static void buffer_set(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    aclrtSetDevice(0);
    aclrtMemcpy(static_cast<uint8_t *>(tensor->data) + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
}
static void buffer_get(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    aclrtSetDevice(0);
    aclrtMemcpy(data, size, static_cast<const uint8_t *>(tensor->data) + offset, size, ACL_MEMCPY_DEVICE_TO_HOST);
}
static void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    aclrtSetDevice(0);
    auto * ctx = static_cast<ascend310_buffer_context *>(buffer->context);
    aclrtMemset(ctx->base, buffer->size, value, buffer->size);
}
static ggml_backend_buffer_t buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * base = nullptr;
    if (!acl_ok(aclrtSetDevice(0), "select device for tensor allocation")) return nullptr;
    if (!acl_ok(aclrtMalloc(&base, size, ACL_MEM_MALLOC_HUGE_FIRST), "allocate tensor buffer")) return nullptr;
    auto * ctx = new ascend310_buffer_context{base};
    const ggml_backend_buffer_i iface = {buffer_free, buffer_base, nullptr, buffer_memset, buffer_set, buffer_get,
        nullptr, nullptr, nullptr, buffer_clear, nullptr, nullptr, nullptr, nullptr};
    return ggml_backend_buffer_init(buft, iface, ctx, size);
}
static const char * buffer_type_name(ggml_backend_buffer_type_t) { return "ASCEND310"; }
static size_t buffer_alignment(ggml_backend_buffer_type_t) { return 64; }
static bool buffer_is_host(ggml_backend_buffer_type_t) { return false; }

static const char * device_name(ggml_backend_dev_t dev) { return static_cast<ascend310_device_context *>(dev->context)->name.c_str(); }
static const char * device_description(ggml_backend_dev_t dev) { return static_cast<ascend310_device_context *>(dev->context)->description.c_str(); }
static void device_memory(ggml_backend_dev_t, size_t * free, size_t * total) { if (free) *free = 0; if (total) *total = 0; }
static enum ggml_backend_dev_type device_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = device_name(dev);
    props->description = device_description(dev);
    props->memory_free = 0;
    props->memory_total = 0;
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    props->caps = {false, false, false, false, false};
}
static ggml_backend_buffer_type_t device_buffer_type(ggml_backend_dev_t) { return &ggml_backend_ascend310_buffer_type; }
static bool device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == &ggml_backend_ascend310_buffer_type; }

static const char * registry_name(ggml_backend_reg_t) { return "ASCEND310"; }
static size_t registry_device_count(ggml_backend_reg_t reg) {
    return static_cast<ascend310_registry_context *>(reg->context)->devices.size();
}
static ggml_backend_dev_t registry_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = static_cast<ascend310_registry_context *>(reg->context);
    return index < ctx->devices.size() ? ctx->devices[index] : nullptr;
}
static void * registry_proc_address(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, "ggml_backend_ascend310_init") == 0) return (void *) ggml_backend_ascend310_init;
    return nullptr;
}

static const ggml_backend_buffer_type_i buffer_type_iface = {buffer_type_name, buffer_alloc, nullptr, buffer_alignment, nullptr, nullptr, nullptr, buffer_is_host};
static const ggml_backend_device_i device_iface = {device_name, device_description, device_memory, device_type, device_props,
    init_backend, device_buffer_type, nullptr, nullptr, supports_op, device_supports_buft, nullptr, nullptr, nullptr, nullptr};
static const ggml_backend_reg_i registry_iface = {registry_name, registry_device_count, registry_device, registry_proc_address};

} // namespace

ggml_backend_reg_t ggml_backend_ascend310_reg(void) {
    static std::mutex init_mutex;
    static bool initialized = false;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (!initialized) {
        if (!prepare_atc_environment(cache_root()) || !acl_ok(aclInit(nullptr), "initialize ACL")) return nullptr;
        if (std::atexit(finalize_acl_runtime) != 0) {
            aclFinalize();
            return nullptr;
        }
        if (!acl_ok(aclrtSetDevice(0), "select device")) return nullptr;
        auto * registry_ctx = new ascend310_registry_context;
        auto * dev_ctx = new ascend310_device_context;
        dev_ctx->device = 0;
        dev_ctx->name = "ASCEND3100";
        dev_ctx->description = "Ascend 310 (CANN legacy ACL)";
        ggml_backend_ascend310_buffer_type = {buffer_type_iface, nullptr, nullptr};
        ggml_backend_dev_t dev = new ggml_backend_device{device_iface, &ggml_backend_ascend310_reg_struct, dev_ctx};
        ggml_backend_ascend310_buffer_type.device = dev;
        registry_ctx->devices.push_back(dev);
        ggml_backend_ascend310_reg_struct = {GGML_BACKEND_API_VERSION, registry_iface, registry_ctx};
        initialized = true;
    }
    return &ggml_backend_ascend310_reg_struct;
}

ggml_backend_t ggml_backend_ascend310_init(int32_t device) {
    ggml_backend_reg_t reg = ggml_backend_ascend310_reg();
    if (!reg || device < 0 || (size_t) device >= registry_device_count(reg)) return nullptr;
    return init_backend(registry_device(reg, (size_t) device), nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_ascend310_reg)
