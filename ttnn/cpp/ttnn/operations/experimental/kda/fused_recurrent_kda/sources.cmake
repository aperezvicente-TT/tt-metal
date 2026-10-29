set(TTNN_OP_EXPERIMENTAL_KDA_FUSED_RECURRENT_KDA_API_HEADERS fused_recurrent_kda.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_FUSED_RECURRENT_KDA_SRCS
    fused_recurrent_kda.cpp
    device/fused_recurrent_kda_device_operation.cpp
    device/fused_recurrent_kda_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_FUSED_RECURRENT_KDA_NANOBIND_SRCS
    fused_recurrent_kda_nanobind.cpp
    ../kda_nanobind.cpp
)
