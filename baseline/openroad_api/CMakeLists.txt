find_package(Protobuf REQUIRED)

set(PROTO_FILES
    proto/net_ordering.proto
)

protobuf_generate_cpp(
    PROTO_SRCS
    PROTO_HDRS
    ${PROTO_FILES}
)

add_library(openroad_api STATIC ${PROTO_SRCS} ${PROTO_HDRS})

target_include_directories(openroad_api
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(openroad_api
    PUBLIC
        protobuf::libprotobuf
)
