PATTERN_DEF_GRAMMAR = r"""
tree: "(" class_spec field_spec* ")"

class_spec: ANY | CLASS ("|" CLASS)*

field_spec: "@" FIELD_NAME ("=" (sequence | value))? capture?

sequence: "[" (value capture?)* (ANY capture?)? "]"

value: tree | var | NONE | ESCAPED_STRING

capture: "->" CAPTURE_KEY

var: "$" CAPTURE_KEY

NONE: "None"

CAPTURE_KEY: ("_"|LCASE_LETTER)* LCASE_LETTER

CLASS: CNAME

FIELD_NAME: CNAME

ANY: "*"

%import common.ESCAPED_STRING
%import common.WS
%import common.CNAME
%import common.LCASE_LETTER

%ignore WS
"""
