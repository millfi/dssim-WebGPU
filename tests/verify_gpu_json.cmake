if(NOT DEFINED JSON_PATH OR JSON_PATH STREQUAL "")
    message(FATAL_ERROR "JSON_PATH is required")
endif()

string(REGEX REPLACE "^\"|\"$" "" JSON_PATH "${JSON_PATH}")

if(NOT EXISTS "${JSON_PATH}")
    message(FATAL_ERROR "JSON output not found: ${JSON_PATH}")
endif()

file(READ "${JSON_PATH}" json_text)

string(JSON status ERROR_VARIABLE status_error GET "${json_text}" status)
if(status_error)
    message(FATAL_ERROR "Missing status in ${JSON_PATH}: ${status_error}")
endif()

if(NOT status STREQUAL "ok")
    message(FATAL_ERROR "Unexpected status in ${JSON_PATH}: ${status}")
endif()

string(JSON score_text ERROR_VARIABLE score_error GET "${json_text}" result score_text)
if(score_error)
    message(FATAL_ERROR "Missing result.score_text in ${JSON_PATH}: ${score_error}")
endif()

foreach(key IN ITEMS
    decode_done_to_score_ms
    create_shader_module_ms
    create_pso_ms
    create_buffer_ms
    write_input_buffer_ms
    create_pipeline_layout_ms
    create_bind_group_ms
    dispatch_and_submit_ms
    readback_ms
    post_process_ms
)
    string(JSON profiling_value ERROR_VARIABLE profiling_error GET "${json_text}" profiling ${key})
    if(profiling_error)
        message(FATAL_ERROR "Missing profiling.${key} in ${JSON_PATH}: ${profiling_error}")
    endif()
endforeach()
