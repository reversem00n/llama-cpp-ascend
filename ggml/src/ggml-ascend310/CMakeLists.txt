if (NOT CANN_INSTALL_DIR AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_INSTALL_DIR "$ENV{ASCEND_TOOLKIT_HOME}")
endif()

if (NOT CANN_INSTALL_DIR)
    message(FATAL_ERROR "ASCEND310 backend requires CANN_INSTALL_DIR or ASCEND_TOOLKIT_HOME")
endif()

set(ASCEND310_CANN_ROOT "${CANN_INSTALL_DIR}")
if (CANN_INSTALL_DIR MATCHES "/fwkacllib$")
    get_filename_component(ASCEND310_CANN_ROOT "${CANN_INSTALL_DIR}" DIRECTORY)
    set(ASCEND310_ACL_ROOT "${CANN_INSTALL_DIR}")
elseif (EXISTS "${CANN_INSTALL_DIR}/fwkacllib/include/acl/acl.h")
    set(ASCEND310_ACL_ROOT "${CANN_INSTALL_DIR}/fwkacllib")
else()
    set(ASCEND310_ACL_ROOT "${CANN_INSTALL_DIR}")
endif()

if (NOT EXISTS "${ASCEND310_ACL_ROOT}/include/acl/acl.h" OR NOT EXISTS "${ASCEND310_ACL_ROOT}/include/acl/acl_op.h")
    message(FATAL_ERROR "ASCEND310 backend requires the legacy ACL operator headers under ${ASCEND310_ACL_ROOT}/include")
endif()

ggml_add_backend_library(ggml-ascend310 ggml-ascend310.cpp)
target_include_directories(ggml-ascend310 PRIVATE "${ASCEND310_ACL_ROOT}/include")
target_link_directories(ggml-ascend310 PUBLIC "${ASCEND310_ACL_ROOT}/lib64" "${ASCEND310_CANN_ROOT}/atc/lib64")
target_link_libraries(ggml-ascend310 PUBLIC
    ascendcl
    acl_op_compiler
    ge_runner
    ge_compiler
    graph
    register
    ge_common
    error_manager
    protobuf
)
target_compile_definitions(ggml-ascend310 PRIVATE GGML_ASCEND310_CANN_ROOT="${ASCEND310_CANN_ROOT}")
target_link_options(ggml-ascend310 PUBLIC
    "-Wl,--disable-new-dtags"
    "-Wl,-rpath,${ASCEND310_ACL_ROOT}/lib64:${ASCEND310_CANN_ROOT}/atc/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common"
)
