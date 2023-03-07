set(model_v1_srcs
        "${CMAKE_CURRENT_SOURCE_DIR}/floor_wetness_v1/floor_wetness_v1.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/floor_wetness_v1/floor_wetness_v1_settings.cc"
        )

set(model_v2_srcs
        "${CMAKE_CURRENT_SOURCE_DIR}/floor_wetness_v2/floor_wetness_v2.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/floor_wetness_v2/floor_wetness_v2_settings.cc"
        )

set(model_v13_srcs
        "${CMAKE_CURRENT_SOURCE_DIR}/floor_wetness_v13/floor_wetness_v13.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/floor_wetness_v13/floor_wetness_v13_settings.cc"
        )

idf_component_register(
    SRCS "${model_v1_srcs}" "${model_v2_srcs}" "${model_v13_srcs}"
        )