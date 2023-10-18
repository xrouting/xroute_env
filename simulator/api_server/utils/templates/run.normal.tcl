set_thread_count {{thread_count}}

read_lef {{input_lef}}
read_def {{input_def}}
read_guides {{input_guides}}

detailed_route \
        -output_drc {{output_drc}} \
        {%- if droute_end_iter > -1 %}
        -droute_end_iter {{droute_end_iter}} \
        {%- endif %}
        -verbose {{verbose}}

write_def {{output_def}}
