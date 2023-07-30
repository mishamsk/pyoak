from __future__ import annotations

import enum
import logging
import re
from itertools import chain
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Generator, Iterator, Sequence, Type, cast

from lark import Lark, Token, Tree, UnexpectedInput
from lark.lexer import Lexer, LexerState
from lark.visitors import Interpreter

from pyoak.match.error import ASTPatternDefinitionError, ASTPatternDidNotMatchError
from pyoak.match.grammar import PATTERN_DEF_GRAMMAR
from pyoak.match.helpers import check_ast_node_type, maybe_capture_rule
from pyoak.node import ASTNode

from ..serialize import TYPES

if TYPE_CHECKING:
    from lark.common import LexerConf
    from lark.parsers.lalr_parser import ParserState

pattern_def_parser = Lark(grammar=PATTERN_DEF_GRAMMAR, start="start", parser="lalr")

INDENT = "    "


logger = logging.getLogger(__name__)


@enum.unique
class Constants(str, enum.Enum):
    ANY_CLASS = "ANY_CLASS"
    ANY_CHILD_FIELD = "ANY_CHILD_FIELD"
    ANY_ATTR_FIELD = "ANY_ATTR_FIELD"
    ANY_ATTR_VALUE = "ANY_ATTR_VALUE"
    ANY_TREE = "ANY_TREE"
    ANY_TREE_ARRAY = "ANY_TREE_ARRAY"
    ANY_ATTR_SPEC_EXCEPT_PREFIX = "any_attr_spec_except_"
    ANY_CHILD_SPEC_EXCEPT_PREFIX = "any_child_spec_except_"
    CLASS_PREFIX = "CLASS_"
    CHILD_FIELD_PREFIX = "CHILD_FIELD_"
    ATTR_FIELD_PREFIX = "ATTR_FIELD_"
    ATTR_VALUE_PREFIX = "ATTR_VALUE_"
    NONE_TOKEN = "NONE_TOKEN"
    EMPTY_TUPLE_TOKEN = "EMPTY_TUPLE_TOKEN"
    CAPTURE_ATTR_RULE_PREFIX = "cap_attr_val_"  # Used to capture the value of an attribute
    CAPTURE_CHILD_RULE_PREFIX = "cap_child_field_"  # Used to capture a child node
    CAPTURE_CHILD_ARRAY_RULE_PREFIX = "cap_child_array_"  # Used to capture a child array
    MATCH_RULE_PREFIX = "match_rule_"
    """Prefix to be used by clients of this module to name match rules."""

    def __str__(self) -> str:
        return self.value


CAPTURE_ATTR_KEY_RE = re.compile(
    Constants.CAPTURE_ATTR_RULE_PREFIX + r"(?P<key>[a-z_]*[a-z])(_\d+)?"
)
CAPTURE_CHILD_KEY_RE = re.compile(
    Constants.CAPTURE_CHILD_RULE_PREFIX + r"(?P<key>[a-z_]*[a-z])(_\d+)?"
)
CAPTURE_CHILD_ARRAY_KEY_RE = re.compile(
    Constants.CAPTURE_CHILD_ARRAY_RULE_PREFIX + r"(?P<key>[a-z_]*[a-z])(_\d+)?"
)


class PatternDefInterpreter(Interpreter[str, str]):
    def __init__(self) -> None:
        super().__init__()

        # Used to store all class names mentioned in rules and generate terminal symbols for them
        self._class_names: dict[str, int] = {}

        # Used to store all attr field names mentioned in rules and generate terminal symbols for them
        self._attr_field_names: dict[str, int] = {}

        # Used to store all child field names mentioned in rules and generate terminal symbols for them
        self._child_field_names: dict[str, int] = {}

        # Used to store all attr values mentioned in rules and generate terminal symbols for them
        self._attr_values: dict[str, int] = {}

        # Used to store all combinations of attr field names that need to be created as except rules
        self._attr_specs_except_rules: dict[tuple[str, ...], int] = {}

        # Used to store all combinations of child field names that need to be created as except rules
        self._child_specs_except_rules: dict[tuple[str, ...], int] = {}

        # Used to store all attr values capture rules (maps rule suffix to actual rule)
        self._capture_attr_rules: dict[str, str] = {}

        # Used to store all child capture rules (maps rule suffix to actual rule)
        self._capture_child_rules: dict[str, str] = {}

        # Used to store all child array capture rules (maps rule suffix to actual rule)
        self._capture_child_array_rules: dict[str, str] = {}

    def start(self, tree: Tree[str]) -> str:
        if len(tree.children) == 1 and isinstance(tree.children[0], Tree):
            rule = self.visit(tree.children[0])
        else:
            raise RuntimeError(f"Unexpected child in start rule: {tree.children}")

        # Start rule is the topmost tree
        out = f"?start: {rule}\n"

        # Specific terminals
        any_class = f"{Constants.ANY_CLASS}"
        for class_name, i in self._class_names.items():
            # Add terminal def for current class
            out += f'{Constants.CLASS_PREFIX}{i}: "{class_name}"\n'
            any_class += f" | {Constants.CLASS_PREFIX}{i}"

        any_attr_field = f"{Constants.ANY_ATTR_FIELD}"
        for attr_field_name, i in self._attr_field_names.items():
            # Add terminal def for current attr field
            out += f'{Constants.ATTR_FIELD_PREFIX}{i}: "{attr_field_name}"\n'
            any_attr_field += f" | {Constants.ATTR_FIELD_PREFIX}{i}"

        any_child_field = f"{Constants.ANY_CHILD_FIELD}"
        for child_field_name, i in self._child_field_names.items():
            # Add terminal def for current child field
            out += f'{Constants.CHILD_FIELD_PREFIX}{i}: "{child_field_name}"\n'
            any_child_field += f" | {Constants.CHILD_FIELD_PREFIX}{i}"

        for attr_value, i in self._attr_values.items():
            # Add terminal def for current attr value
            inner = attr_value[1:-1].replace('\\"', '"')
            out += f"{Constants.ATTR_VALUE_PREFIX}{i}: /{inner}/\n"

        # Except rules for attr specs
        for exclude_attr_specs, i in self._attr_specs_except_rules.items():
            allowed_attr_field_token_names = [f"{Constants.ANY_ATTR_FIELD}"]
            for attr_field_name, j in self._attr_field_names.items():
                if attr_field_name not in exclude_attr_specs:
                    allowed_attr_field_token_names.append(f"{Constants.ATTR_FIELD_PREFIX}{j}")

            out += f'{Constants.ANY_ATTR_SPEC_EXCEPT_PREFIX}{i}: ({"|".join(allowed_attr_field_token_names)}) "=" any_attr_value\n'

        # Except rules for child specs
        for exclude_child_specs, i in self._child_specs_except_rules.items():
            allowed_child_field_token_names = [f"{Constants.ANY_CHILD_FIELD}"]
            for child_field_name, j in self._child_field_names.items():
                if child_field_name not in exclude_child_specs:
                    allowed_child_field_token_names.append(f"{Constants.CHILD_FIELD_PREFIX}{j}")

            out += f'{Constants.ANY_CHILD_SPEC_EXCEPT_PREFIX}{i}: ({"|".join(allowed_child_field_token_names)}) "=" ({Constants.ANY_TREE_ARRAY} | {Constants.ANY_TREE} | {Constants.NONE_TOKEN} | {Constants.EMPTY_TUPLE_TOKEN})\n'

        # Capture attr values rules
        for rule_name, rule in self._capture_attr_rules.items():
            out += f"{Constants.CAPTURE_ATTR_RULE_PREFIX}{rule_name}: {rule}\n"

        # Capture child rules
        for rule_name, rule in self._capture_child_rules.items():
            out += f"{Constants.CAPTURE_CHILD_RULE_PREFIX}{rule_name}: {rule}\n"

        # Capture child array rules
        for rule_name, rule in self._capture_child_array_rules.items():
            out += f"{Constants.CAPTURE_CHILD_ARRAY_RULE_PREFIX}{rule_name}: {rule}\n"

        # Any terminals and rules
        out += f"any_class: {any_class}\n"
        out += 'any_child_spec: any_child_field "=" any_child_field_value\n'
        out += f"any_child_field: {any_child_field}\n"
        out += f"any_child_field_value: {Constants.ANY_TREE_ARRAY} | {Constants.ANY_TREE} | {Constants.NONE_TOKEN} | {Constants.EMPTY_TUPLE_TOKEN}\n"
        out += 'any_attr_spec: any_attr_field "=" any_attr_value\n'
        out += f"any_attr_field: {any_attr_field}\n"
        out += f"any_attr_value: {Constants.ANY_ATTR_VALUE} | {Constants.NONE_TOKEN}\n"

        # Special error token for lexer
        out += f"%declare {Constants.ANY_CLASS} {Constants.ANY_TREE} {Constants.ANY_TREE_ARRAY} {Constants.ANY_CHILD_FIELD} {Constants.ANY_ATTR_FIELD} {Constants.ANY_ATTR_VALUE} {Constants.NONE_TOKEN} {Constants.EMPTY_TUPLE_TOKEN}"

        logger.debug("Generated tree matching grammar:\n%s", out)

        return out

    def tree(self, tree: Tree[str]) -> str:
        assert isinstance(tree.children[0], str)
        class_name = tree.children[0]

        if class_name == "*":
            class_name = Constants.ANY_CLASS
        elif class_name not in self._class_names:
            ok, msg = check_ast_node_type(class_name)
            if not ok:
                raise ASTPatternDefinitionError(msg)

            self._class_names[class_name] = i = len(self._class_names)
            class_name = f"{Constants.CLASS_PREFIX}{i}"
        else:
            class_name = f"{Constants.CLASS_PREFIX}{self._class_names[class_name]}"

        out = '"(" ' + class_name

        attr_specs = ""
        child_specs = ""
        for child in tree.children[1:]:
            if isinstance(child, Tree):
                if child.data == "attr_specs":
                    attr_specs = self.visit(child)
                elif child.data == "child_specs":
                    child_specs = self.visit(child)
            else:
                raise RuntimeError(f"Unexpected child in tree rule: {child}")

        # No attr specs
        if attr_specs == "":
            out += " (any_attr_spec)*"
        else:
            out += " " + attr_specs

        # No child specs
        if child_specs == "":
            out += " (any_child_spec)*"
        else:
            out += " " + child_specs

        out += ' ")"'
        return out

    def attr_specs(self, tree: Tree[str]) -> str:
        out = ""
        specs: list[str] = []
        spec_field_name_to_spec: dict[str, str] = {}
        for child in tree.children:
            if isinstance(child, Tree):
                one_spec = self.visit(child)
                specs.append(one_spec)

                # This must be the attr field name
                assert isinstance(child.children[0], str)
                spec_field_name_to_spec[child.children[0]] = one_spec

        # Grammar should not allow empty attr specs
        assert len(specs) > 0

        all_ = False
        only = False
        if isinstance((tok := tree.children[-1]), Token):
            if tok.type == "ALL_IN_ORDER":
                all_ = True
            elif tok.type == "ONLY":
                only = True

        if not all_ and not only:
            # Allow any other attrs with arbitrary values, and some of the defined attrs
            # In order to do that we need to create "except" rule that will capture anything
            # that is not in the defined attr specs
            sorted_spec_field_names = tuple(sorted(spec_field_name_to_spec.keys()))

            except_rule_index = -1
            if sorted_spec_field_names in self._attr_specs_except_rules:
                except_rule_index = self._attr_specs_except_rules[sorted_spec_field_names]
            else:
                except_rule_index = len(self._attr_specs_except_rules)
                self._attr_specs_except_rules[sorted_spec_field_names] = except_rule_index

            # What we want is to have any other attributes, but if we hit the one
            # defined in the spec it must match exactly. So we need to create a long
            # rule which alternates between any other attr and the defined ones.
            any_other_subrule = f"({Constants.ANY_ATTR_SPEC_EXCEPT_PREFIX}{except_rule_index})*"

            for spec in specs:
                out += any_other_subrule + " " + spec + " "

            # Also account for potential trailing any other attrs
            out += any_other_subrule
        elif all_:
            out += " ".join(map(itemgetter(1), sorted(spec_field_name_to_spec.items())))
        else:
            out += "(" + "|".join(specs) + ")+"

        return out

    def child_specs(self, tree: Tree[str]) -> str:
        out = ""
        specs: list[str] = []
        spec_field_name_to_spec: dict[str, str] = {}
        for child in tree.children:
            if isinstance(child, Tree):
                one_spec = self.visit(child)
                specs.append(one_spec)

                # This must be the attr field name
                assert isinstance(child.children[0], str)
                spec_field_name_to_spec[child.children[0]] = one_spec

        # Grammar should not allow empty child specs
        assert len(specs) > 0

        all_ = False
        only = False
        if isinstance((tok := tree.children[-1]), Token):
            if tok.type == "ALL_IN_ORDER":
                all_ = True
            elif tok.type == "ONLY":
                only = True

        if not all_ and not only:
            # Allow any other children with arbitrary subtrees, and some of the defined children
            # In order to do that we need to create "except" rule that will capture anything
            # that is not in the defined child specs
            sorted_spec_field_names = tuple(sorted(spec_field_name_to_spec.keys()))

            except_rule_index = -1
            if sorted_spec_field_names in self._child_specs_except_rules:
                except_rule_index = self._child_specs_except_rules[sorted_spec_field_names]
            else:
                except_rule_index = len(self._child_specs_except_rules)
                self._child_specs_except_rules[sorted_spec_field_names] = except_rule_index

            # What we want is to have any other attributes, but if we hit the one
            # defined in the spec it must match exactly. So we need to create a long
            # rule which alternates between any other attr and the defined ones.
            any_other_subrule = f"({Constants.ANY_CHILD_SPEC_EXCEPT_PREFIX}{except_rule_index})*"

            for spec in specs:
                out += any_other_subrule + " " + spec + " "

            # Also account for potential trailing any other attrs
            out += any_other_subrule
        elif all_:
            out += " ".join(map(itemgetter(1), sorted(spec_field_name_to_spec.items())))
        else:
            out += "(" + "|".join(specs) + ")+"

        return out

    def attr_spec(self, tree: Tree[str]) -> str:
        assert isinstance(tree.children[0], str)
        attr_field_name = tree.children[0]

        if attr_field_name not in self._attr_field_names:
            self._attr_field_names[attr_field_name] = i = len(self._attr_field_names)
            attr_field_name = f"{Constants.ATTR_FIELD_PREFIX}{i}"
        else:
            attr_field_name = (
                f"{Constants.ATTR_FIELD_PREFIX}{self._attr_field_names[attr_field_name]}"
            )

        value = "any_attr_value"
        if len(tree.children) > 1:
            if isinstance((sub := tree.children[1]), Tree) and sub.data != "capture":
                # attr_value subrule
                value = self.visit(sub)
            elif isinstance(sub, Token):
                # must be NONE token
                value = f"{Constants.NONE_TOKEN}"

        out = f'{attr_field_name} "=" {value}'

        if len(tree.children) > 1:
            out = maybe_capture_rule(
                maybe_capture_child=tree.children[-1],
                inner_rule=out,
                capture_rule_prefix=Constants.CAPTURE_ATTR_RULE_PREFIX,
                capture_rules=self._capture_attr_rules,
            )

        return out

    def attr_values(self, tree: Tree[str]) -> str:
        return "(" + "|".join(self.visit(cast(Tree[str], child)) for child in tree.children) + ")"

    def attr_value(self, tree: Tree[str]) -> str:
        assert isinstance(tree.children[0], str)
        attr_value = tree.children[0]

        if attr_value not in self._attr_values:
            self._attr_values[attr_value] = i = len(self._attr_values)
            attr_value = f"{Constants.ATTR_VALUE_PREFIX}{i}"
        else:
            attr_value = f"{Constants.ATTR_VALUE_PREFIX}{self._attr_values[attr_value]}"

        return attr_value

    def child_spec(self, tree: Tree[str]) -> str:
        assert isinstance(tree.children[0], str)
        child_field_name = tree.children[0]

        if child_field_name not in self._child_field_names:
            self._child_field_names[child_field_name] = i = len(self._child_field_names)
            child_field_name = f"{Constants.CHILD_FIELD_PREFIX}{i}"
        else:
            child_field_name = (
                f"{Constants.CHILD_FIELD_PREFIX}{self._child_field_names[child_field_name]}"
            )

        out = f"{child_field_name}"

        child_tree = "any_child_field_value"
        if len(tree.children) > 1:
            if isinstance((sub := tree.children[1]), Tree) and sub.data != "capture":
                # tree_array or tree subrule
                child_tree = self.visit(sub)
            elif isinstance(sub, Token):
                if sub.type == "NONE":
                    child_tree = f"{Constants.NONE_TOKEN}"
                elif sub.type == "EMPTY_TUPLE":
                    child_tree = f"{Constants.EMPTY_TUPLE_TOKEN}"

        out = f'{child_field_name} "=" {child_tree}'

        if len(tree.children) > 1:
            out = maybe_capture_rule(
                maybe_capture_child=tree.children[-1],
                inner_rule=out,
                capture_rule_prefix=Constants.CAPTURE_CHILD_RULE_PREFIX,
                capture_rules=self._capture_child_rules,
            )

        return out

    def tree_array(self, tree: Tree[str]) -> str:
        out = '"[" '
        for c in tree.children:
            if isinstance(c, Tree):
                out += self.visit(c) + " "
            elif isinstance(c, Token):
                # Must be ANY token
                assert c.type == "ANY"
                any_rule = f"{Constants.ANY_TREE_ARRAY}"

                any_rule = maybe_capture_rule(
                    maybe_capture_child=tree.children[-1],
                    inner_rule=any_rule,
                    capture_rule_prefix=Constants.CAPTURE_CHILD_RULE_PREFIX,
                    capture_rules=self._capture_child_rules,
                )

                out += f" {any_rule}"
                break

        out += ' "]"'

        return out

    def tree_array_seq(self, tree: Tree[str]) -> str:
        out = ""
        trees_and_counts: list[tuple[str, int | None]] = []
        all_in_order = False
        capture_child: Tree[str] | None = None
        for child in tree.children:
            if isinstance(child, Tree) and child.data == "tree":
                trees_and_counts.append((self.visit(child), None))
            elif isinstance(child, Tree) and child.data == "capture":
                capture_child = child
            elif isinstance(child, Token):
                if child.type == "COUNT":
                    trees_and_counts[-1] = (trees_and_counts[-1][0], int(child.value))
                elif child.type == "ALL_IN_ORDER":
                    all_in_order = True

        # TODO: The mode without count and without ALL_IN_ORDER must be removed as this
        # leads to ambiguity if any other tree_array_seq is following.
        if any(count is not None for _, count in trees_and_counts):
            # Force all_in_order if any count is specified
            all_in_order = True

        if all_in_order:
            for tree_, count in trees_and_counts:
                if count is None:
                    out += f"{tree_} "
                else:
                    out += f"({tree_})~{count} "
        else:
            out += "(" + "|".join(tree_ for tree_, _ in trees_and_counts) + ")* "

        out = maybe_capture_rule(
            maybe_capture_child=capture_child,
            inner_rule=out,
            capture_rule_prefix=Constants.CAPTURE_CHILD_ARRAY_RULE_PREFIX,
            capture_rules=self._capture_child_array_rules,
        )

        return out


class ASTTokenType(enum.Enum):
    DELIM = enum.auto()
    CLASS_INSTANCE = enum.auto()
    ATTR_FIELD = enum.auto()
    ATTR_VALUE = enum.auto()
    CHILD_FIELD = enum.auto()
    CHILD_VALUE = enum.auto()


class ASTToken(Token):
    __slots__ = ("ast_type", "ast_nodes")

    ast_type: ASTTokenType
    ast_nodes: Sequence[ASTNode]

    def __new__(
        cls,
        type: str,
        value: Any,
        ast_type: ASTTokenType,
        ast_nodes: Sequence[ASTNode] | None = None,
    ) -> ASTToken:
        inst = str.__new__(cls)

        if ast_nodes is None:
            ast_nodes = []

        # lark standard fields
        inst.type = type
        inst.value = value
        inst.start_pos = None
        inst.line = None
        inst.column = None
        inst.end_line = None
        inst.end_column = None
        inst.end_pos = None

        # ast extension fields
        inst.ast_type = ast_type
        inst.ast_nodes = ast_nodes

        return inst

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.type=}, {self.value=}, {self.ast_type})"

    def __str__(self) -> str:
        return repr(self)


class ASTMatchingLexer(Lexer):
    __future_interface__ = True

    def __init__(self, lexer_conf: LexerConf) -> None:
        self._terminal_to_ast_class: dict[str, Type[ASTNode]] = {}
        self._attr_field_name_to_terminal: dict[str, str] = {}
        self._child_field_name_to_terminal: dict[str, str] = {}

        self._terminals = lexer_conf.terminals
        self._attr_value_terminal_to_regex: dict[str, re.Pattern[str]] = {}

        for term_def in lexer_conf.terminals:
            if term_def.name.startswith(Constants.CLASS_PREFIX):
                class_name = term_def.pattern.value
                if class_name is None:
                    raise RuntimeError(f"Missing class name for {term_def.name}")

                # We've already checked that the class name is a valid ASTNode subclass, sso casting
                self._terminal_to_ast_class[term_def.name] = cast(
                    Type[ASTNode], TYPES.get(class_name, ASTNode)
                )
            elif term_def.name.startswith(Constants.ATTR_FIELD_PREFIX):
                self._attr_field_name_to_terminal[term_def.pattern.value] = term_def.name
            elif term_def.name.startswith(Constants.ATTR_VALUE_PREFIX):
                # Remove quotes from value
                self._attr_value_terminal_to_regex[term_def.name] = re.compile(
                    term_def.pattern.to_regexp()
                )
            elif term_def.name.startswith(Constants.CHILD_FIELD_PREFIX):
                self._child_field_name_to_terminal[term_def.pattern.value] = term_def.name

        self._terminal_to_ast_class[Constants.ANY_CLASS] = ASTNode

        self._cname_re = re.compile(r"^[_a-z](\w+)", flags=re.IGNORECASE)

    def lex(self, lexer_state: LexerState, parser_state: ParserState) -> Iterator[Token]:
        return self._feed_ast_node(cast(ASTNode, lexer_state.text), parser_state)

    def _feed_ast_node(
        self, node: ASTNode, parser_state: ParserState
    ) -> Generator[Token, None, None]:
        # First the grammar expects a LPAR
        yield ASTToken("LPAR", "(", ASTTokenType.DELIM, ast_nodes=[node])

        # then a CLASS_* or ANY_CLASS for the current node
        expected_class: Type[ASTNode] | None = None
        for expected in parser_state.parse_conf.states[parser_state.position]:
            if expected.startswith(Constants.CLASS_PREFIX) or expected == Constants.ANY_CLASS:
                expected_class = self._terminal_to_ast_class[expected]
                if isinstance(node, expected_class):
                    yield ASTToken(expected, node, ASTTokenType.CLASS_INSTANCE, ast_nodes=[node])
                    break
        else:
            if expected_class is not None:
                raise ASTPatternDidNotMatchError(
                    f"Expected {expected_class.__name__} but got {node.__class__.__name__}"
                )
            else:
                raise ASTPatternDidNotMatchError("Unexpected grammar state")

        # then a list of ATTR_FIELD_* and ATTR_VALUE_* for the current node
        for val, f in sorted(
            node.get_properties(
                skip_id=False,
                skip_origin=False,
                skip_original_id=False,
                skip_id_collision_with=False,
                skip_hidden=False,
                skip_non_compare=False,
            ),
            key=lambda x: x[1].name,
        ):
            if f.name in self._attr_field_name_to_terminal:
                yield ASTToken(
                    self._attr_field_name_to_terminal[f.name],
                    f.name,
                    ASTTokenType.ATTR_FIELD,
                    ast_nodes=[node],
                )
            else:
                yield ASTToken(
                    Constants.ANY_ATTR_FIELD,
                    f.name,
                    ASTTokenType.ATTR_FIELD,
                    ast_nodes=[node],
                )

            yield ASTToken("EQUAL", "=", ASTTokenType.DELIM, ast_nodes=[node])

            if val is None:
                yield ASTToken(
                    Constants.NONE_TOKEN,
                    val,
                    ASTTokenType.ATTR_VALUE,
                    ast_nodes=[node],
                )
            else:
                for expected in parser_state.parse_conf.states[parser_state.position]:
                    if expected.startswith(Constants.ATTR_VALUE_PREFIX):
                        attr_value_re = self._attr_value_terminal_to_regex[expected]
                        if attr_value_re.match(str(val)) is not None:
                            yield ASTToken(
                                expected,
                                val,
                                ASTTokenType.ATTR_VALUE,
                                ast_nodes=[node],
                            )
                            break
                else:
                    yield ASTToken(
                        Constants.ANY_ATTR_VALUE,
                        val,
                        ASTTokenType.ATTR_VALUE,
                        ast_nodes=[node],
                    )

        for child, f in sorted(node._iter_child_fields(), key=lambda x: x[1].name):
            if isinstance(child, ASTNode):
                ast_nodes: Sequence[ASTNode] | None = [child]
            else:
                ast_nodes = child

            if f.name in self._child_field_name_to_terminal:
                yield ASTToken(
                    self._child_field_name_to_terminal[f.name],
                    f.name,
                    ASTTokenType.CHILD_FIELD,
                    ast_nodes=ast_nodes,
                )
            else:
                yield ASTToken(
                    Constants.ANY_CHILD_FIELD,
                    f.name,
                    ASTTokenType.CHILD_FIELD,
                    ast_nodes=ast_nodes,
                )

            yield ASTToken("EQUAL", "=", ASTTokenType.DELIM, ast_nodes=[node])

            if child is None:
                yield ASTToken(
                    Constants.NONE_TOKEN,
                    None,
                    ASTTokenType.CHILD_VALUE,
                    ast_nodes=None,
                )
            elif isinstance(child, Sequence):
                if len(child) == 0:
                    yield ASTToken(
                        Constants.EMPTY_TUPLE_TOKEN,
                        None,
                        ASTTokenType.CHILD_VALUE,
                        ast_nodes=child,
                    )
                else:
                    # If we expect any tree array, we just output a token, no need to go deeper
                    if (
                        Constants.ANY_TREE_ARRAY
                        in parser_state.parse_conf.states[parser_state.position]
                    ):
                        # We've already checked that the class name is a valid ASTNode subclass, sso casting
                        yield ASTToken(
                            Constants.ANY_TREE_ARRAY,
                            child,
                            ASTTokenType.CHILD_VALUE,
                            ast_nodes=child,
                        )
                    else:
                        # If not any, we need to iterate over children

                        # yield open bracket
                        yield ASTToken("LSQB", "[", ASTTokenType.DELIM, ast_nodes=[node])

                        for i, c in enumerate(child):
                            # If we expect any trailing tree array, we just output a token and break
                            if (
                                Constants.ANY_TREE_ARRAY
                                in parser_state.parse_conf.states[parser_state.position]
                            ):
                                # Any tree array may only appear last => we attach all remaining children
                                yield ASTToken(
                                    Constants.ANY_TREE_ARRAY,
                                    child,
                                    ASTTokenType.CHILD_VALUE,
                                    ast_nodes=child[i:],
                                )
                                break

                            yield from self._feed_ast_node(c, parser_state)

                        # yield close bracket
                        yield ASTToken("RSQB", "]", ASTTokenType.DELIM, ast_nodes=[node])
            else:
                # If we expect any tree, we just output a token, no need to go deeper
                if Constants.ANY_TREE in parser_state.parse_conf.states[parser_state.position]:
                    yield ASTToken(
                        Constants.ANY_TREE,
                        child,
                        ASTTokenType.CHILD_VALUE,
                        ast_nodes=[child],
                    )
                else:
                    yield from self._feed_ast_node(child, parser_state)

        yield ASTToken("RPAR", ")", ASTTokenType.DELIM, ast_nodes=[node])


class MatchExtractor(Interpreter[ASTToken, dict[str, list[Any]]]):
    def __default__(self, tree: Tree[ASTToken]) -> dict[str, list[Any]]:
        if isinstance(tree.data, str):
            if (m := CAPTURE_ATTR_KEY_RE.match(tree.data)) is not None:
                key = m["key"]
                assert len(tree.children) == 2
                val_item = tree.children[1]
                if isinstance(val_item, ASTToken):
                    return {key: [val_item.value]}
                elif isinstance(val_item, Tree):
                    assert len(val_item.children) == 1
                    val_item = val_item.children[0]
                    assert isinstance(val_item, ASTToken)
                    return {key: [val_item.value]}
            elif (m := CAPTURE_CHILD_KEY_RE.match(tree.data)) is not None:
                key = m["key"]
                assert len(tree.children) > 0
                child_field_token = tree.children[0]
                if isinstance(child_field_token, ASTToken):
                    return {key: list(child_field_token.ast_nodes)}
            elif (m := CAPTURE_CHILD_ARRAY_KEY_RE.match(tree.data)) is not None:
                key = m["key"]
                out_nodes: list[ASTNode] = []
                for c in tree.children:
                    if isinstance(c, ASTToken) and (
                        c.type.startswith(Constants.CLASS_PREFIX)
                        or c.type.startswith(Constants.ANY_CLASS)
                    ):
                        # Every CLASS_xxx means a new child in array
                        out_nodes.extend(c.ast_nodes)
                return {key: out_nodes}

        ret: dict[str, list[Any]] = {}
        for c in tree.children:
            if isinstance(c, Tree):
                ch_ret = self.visit(c)
                ret = {
                    k: ret.get(k, []) + ch_ret.get(k, [])
                    for k in set(chain(ret.keys(), ch_ret.keys()))
                }

        return ret


class PatternMatcher:
    def __init__(self, pattern_defs: Sequence[tuple[str, str]]) -> None:
        """A tree pattern matcher.

        Args:
            pattern_defs (Sequence[tuple[str, str]]): A sequence of tuples of pattern name and pattern definition.
                Pattern names must be unqiue and are used identify the pattern in the match result.


        Raises:
            ASTPatternDefinitionError: Raised if the pattern definition is incorrect

        """

        if len({pd[0] for pd in pattern_defs}) != len(pattern_defs):
            raise ASTPatternDefinitionError("Pattern names must be unique")

        self._all_rule_names = [pd[0] for pd in pattern_defs]

        incorrect_patterns: list[tuple[str, str]] = []
        parsed_pattern_defs: list[tuple[str, Tree[str]]] = []
        for pattern_name, pattern_def in pattern_defs:
            try:
                parsed_pattern_defs.append(
                    (
                        pattern_name,
                        cast(Tree[str], pattern_def_parser.parse(pattern_def)),
                    )
                )
            except UnexpectedInput as e:  # noqa: PERF203
                incorrect_patterns.append(
                    (
                        pattern_name,
                        f"Error at:\n{e.get_context(pattern_def)}",
                    )
                )
            except Exception as e:
                incorrect_patterns.append(
                    (
                        pattern_name,
                        "Unknown error.",
                    )
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Unexpected error during pattern definition parsing: {e}")

        self._match_grammars: dict[str, str] = {}
        self._match_parsers: dict[str, Lark] = {}

        for pattern_name, parsed_pattern_def in parsed_pattern_defs:
            try:
                match_grammar = PatternDefInterpreter().visit(parsed_pattern_def)

                self._match_grammars[pattern_name] = match_grammar

                self._match_parsers[pattern_name] = Lark(
                    grammar=match_grammar,
                    parser="lalr",
                    lexer=ASTMatchingLexer,
                )

            except ASTPatternDefinitionError as e:  # noqa: PERF203
                incorrect_patterns.append((pattern_name, e.message))
            except Exception as e:
                incorrect_patterns.append((pattern_name, "Unknown error"))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Unexpected error during pattern definition grammar generation: {e}"
                    )

        if incorrect_patterns:
            raise ASTPatternDefinitionError(
                "Incorrect pattern definitions:\n"
                + "\n".join(
                    [
                        f"Pattern '{pattern_name}': {pattern_def}"
                        for pattern_name, pattern_def in incorrect_patterns
                    ]
                )
            )

        self._match_extractor = MatchExtractor()

    @staticmethod
    def validate_pattern(pattern_def: str) -> tuple[bool, str]:
        """Validate a single pattern definition in a form of `pattern`.

        Args:
            pattern_def: The pattern definition to validate.

        Returns:
            A tuple of a boolean indicating whether the pattern definition is valid and a string
            containing the error message if the pattern definition is invalid.

        """
        try:
            parsed_pattern_defs = pattern_def_parser.parse(pattern_def)
        except UnexpectedInput as e:
            return (
                False,
                f"Incorrect pattern definition. Context:\n{e.get_context(pattern_def)}",
            )
        except Exception:
            return False, "Incorrect pattern definition. Unexpected error"

        try:
            _ = PatternDefInterpreter().visit(cast(Tree[str], parsed_pattern_defs))
        except ASTPatternDefinitionError as e:
            return False, e.message
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Unexpected error during pattern definition grammar generation: {e}")
            return False, "Incorrect pattern definition. Unexpected error"

        return True, "Valid pattern definition"

    def match(
        self, node: ASTNode, rules: list[str] | None = None
    ) -> tuple[str, dict[str, list[Any]]] | None:
        """Match a node against the pattern definitions.

        Args:
            node: The node to match against the pattern definitions.
            rules: The rules to match against (in order). If None, all rules will be checked.

        Returns:
            A tuple of the matched rule name and a dictionary of the matched values.
            If no rule matches, None is returned.

        """
        if rules is None:
            rules = self._all_rule_names

        for rule in rules:
            try:
                res = cast(Tree[ASTToken], self._match_parsers[rule].parse(node))  # type: ignore
                return rule, self._match_extractor.visit(res)
            except UnexpectedInput as e:  # noqa: PERF203
                if logger.isEnabledFor(logging.DEBUG):
                    vals = " ".join(
                        [
                            val.type if isinstance(val, ASTToken) else val.data
                            for val in e.state.value_stack
                        ]
                    )
                    logger.debug(
                        f"Tree didn't match rule {rule}.\n Matched: {vals}\nExpected {e.expected} but got {e.token.type} instead"  # type: ignore
                    )
            except ASTPatternDidNotMatchError as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Tree didn't match rule {rule}.\n{e.message}")

        return None
