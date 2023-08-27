PATTERN_DEF_GRAMMAR = r"""
start: tree

tree: "(" CLASS attr_specs? child_specs? ")"

child_specs: "@[" (child_spec)+ (ONLY | ALL_IN_ORDER)? "]"

child_spec: FIELD_NAME ("=" (tree_array | tree | NONE | EMPTY_TUPLE))? capture?

tree_array: "[" tree_array_seq ("," tree_array_seq)* ("," ANY capture?)? "]"

// The mode without count and without ALL_IN_ORDER must be removed as this
// leads to ambiguity if any other tree_array_seq is following.
tree_array_seq: (tree ("~" COUNT)?)+ ALL_IN_ORDER? capture?

attr_specs: "#[" (attr_spec)+ (ONLY | ALL_IN_ORDER)? "]"

attr_spec: FIELD_NAME ("=" (attr_value | attr_values | NONE))? capture?

attr_values: "(" attr_value ("|" attr_value)* ")"

attr_value: ESCAPED_STRING

capture: "->" CAPTURE_KEY

NONE: "None"

EMPTY_TUPLE: "[]"

COUNT: DIGIT+

CAPTURE_KEY: ("_"|LCASE_LETTER)* LCASE_LETTER

RULE_NAME.2: ("_"|LCASE_LETTER) ("_"|LCASE_LETTER|DIGIT)*

CLASS: CNAME | ANY

FIELD_NAME: CNAME

ANY: "*"

ALL_IN_ORDER: "!!"

ONLY: "!"

%import common.ESCAPED_STRING
%import common.WS
%import common.CNAME
%import common.LCASE_LETTER
%import common.DIGIT

%ignore WS
"""
