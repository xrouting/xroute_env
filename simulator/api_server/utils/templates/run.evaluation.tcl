set_thread_count {{thread_count}}

read_lef {{input_lef}}
read_def {{input_def}}
read_guides {{input_guides}}

detailed_route_debug \
        {%- if custom_strategies %}
        -custom_strategies
        -custom_size {{custom_size}} \
        -custom_offset {{custom_offset}} \
        {%- endif %}
        {%- if parallel_workers %}
        -parallel_workers {{parallel_workers}} \
        {%- endif %}
        {%- if api_address %}
        -api_addr {{api_address}} \
        {%- endif %}
        {%- if api_timeout %}
        -api_timeout {{api_timeout}} \
        {%- endif %}
        -net_ordering_evaluation {{net_ordering_evaluation_mode}}

detailed_route \
        -output_drc {{output_drc}} \
        {%- if droute_end_iter > -1 %}
        -droute_end_iter {{droute_end_iter}} \
        {%- endif %}
        -verbose {{verbose}}

write_def {{output_def}}
