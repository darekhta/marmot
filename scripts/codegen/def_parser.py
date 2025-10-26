#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from codegen_base import strip_comments

# =============================================================================
# Public data model
# =============================================================================

CORE_FIELDS = {
    "ACTIVATION",
    "OP",
    "PROFILE",
    "MATMUL_LAYOUT",
    "PROFILES",
    "PLATFORM",
    "IMPL_FUNCTION",
    "INPUT_DTYPE",
    "WEIGHT_DTYPE",
    "OUTPUT_DTYPE",
    "ACCUM_DTYPE",
    "WEIGHT_QUANT",
    "QUANT_BLOCK",
    "WEIGHT_LAYOUT",
    "STRIDE_MODE",
    "EPILOGUE",
    "BIAS",
    "FUSION",
    "TILING",
    "PIPELINE",
    "COST_MODEL",
    "SHARDABLE",
    "DEVICE_AFFINITY",
    "KERNEL_FUNCTION",
    "THREADGROUP_SIZE",
    "SIMD_GROUP",
    "MPS_FALLBACK",
}


_ELEMENTWISE_BINARY_OPS = {
    "add",
    "mul",
    "sub",
    "div",
    "min",
    "max",
    "pow",
    "mod",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_shl",
    "bitwise_shr",
    "bitwise_shr_logical",
    "compare_eq",
    "compare_ne",
    "compare_lt",
    "compare_le",
    "compare_gt",
    "compare_ge",
}

_UNARY_OPS = {
    "abs",
    "neg",
    "sign",
    "sqrt",
    "exp",
    "log",
    "bitwise_not",
    "relu",
    "gelu",
    "gelu_tanh",
    "silu",
    "sigmoid",
    "tanh",
    "mish",
    "elu",
    "selu",
    "leaky_relu",
    "prelu",
}

_REDUCTION_OPS = {
    "reduction_sum",
    "reduction_mean",
    "reduction_max",
    "reduction_min",
    "reduction_variance",
    "reduction_std",
    "reduction_norm_l1",
    "reduction_norm_l2",
    "reduction_prod",
    "reduction_argmax",
    "reduction_argmin",
    "reduction_any",
    "reduction_all",
}

_LAYOUT_OPS = {
    "reshape",
    "transpose",
    "concat",
    "split",
    "slice",
}

_ARITHMETIC_OPS = _ELEMENTWISE_BINARY_OPS | _UNARY_OPS | {
    "softmax",
    "matmul",
    "qkv_rope",
    "rope",
    "layernorm",
    "rms_norm",
    "vec_dot",
}


DTYPE_SHORT_MAP: Dict[str, str] = {
    "FLOAT16": "f16",
    "FLOAT32": "f32",
    "FLOAT64": "f64",
    "BFLOAT16": "bf16",
    "FLOAT8_E4M3": "fp8_e4m3",
    "FLOAT8_E5M2": "fp8_e5m2",
    "INT8": "i8",
    "INT16": "i16",
    "INT32": "i32",
    "INT64": "i64",
    "UINT8": "u8",
    "UINT16": "u16",
    "UINT32": "u32",
    "UINT64": "u64",
}


@dataclasses.dataclass
class KernelDescriptor:
    name: str
    core_fields: Dict[str, Any]
    extensions: Dict[str, Any]
    path: Path
    source: str
    locations: Dict[str, int] = dataclasses.field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.core_fields.get(key, default)


class DefParseError(RuntimeError):
    def __init__(self, message: str, *, path: Path | None = None, line: int | None = None, column: int | None = None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.line = line
        self.column = column

    def __str__(self) -> str:  # pragma: no cover
        location = ""
        if self.path is not None:
            location = str(self.path)
            if self.line is not None:
                location += f":{self.line}"
                if self.column is not None:
                    location += f":{self.column}"
        if location:
            return f"{location}: {self.message}"
        return self.message


def _offset_to_line_col(text: str, offset: int) -> Tuple[int, int]:
    if offset < 0:
        return 1, 1
    line = text.count("\n", 0, offset) + 1
    last_newline = text.rfind("\n", 0, offset)
    column = offset + 1 if last_newline == -1 else offset - last_newline
    return line, column


def _raise_with_location(message: str, *, path: Path, content: str, offset: int) -> None:
    line, column = _offset_to_line_col(content, offset)
    raise DefParseError(message, path=path, line=line, column=column)


# =============================================================================
# Tokenizer
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Token:
    kind: str  # ident | string | number | symbol | eof
    value: str
    offset: int


_SYMBOLS = {"(", ")", "{", "}", "[", "]", ":", ",", ".", "@", "$", "*"}


def _unescape_string(raw: str) -> str:
    out: List[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= len(raw):
            out.append("\\")
            break
        esc = raw[i]
        if esc == "n":
            out.append("\n")
        elif esc == "t":
            out.append("\t")
        elif esc == "r":
            out.append("\r")
        elif esc == '"':
            out.append('"')
        elif esc == "\\":
            out.append("\\")
        else:
            out.append(esc)
        i += 1
    return "".join(out)


def _tokenize(path: Path, content: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    while i < len(content):
        ch = content[i]
        if ch.isspace():
            i += 1
            continue
        if ch in _SYMBOLS:
            tokens.append(Token("symbol", ch, i))
            i += 1
            continue
        if ch == '"':
            start = i
            i += 1
            buf: List[str] = []
            while i < len(content):
                if content[i] == '"' and (i == 0 or content[i - 1] != "\\"):
                    break
                buf.append(content[i])
                i += 1
            if i >= len(content) or content[i] != '"':
                _raise_with_location("Unterminated string literal", path=path, content=content, offset=start)
            tokens.append(Token("string", _unescape_string("".join(buf)), start))
            i += 1
            continue
        if ch.isdigit():
            start = i
            i += 1
            while i < len(content) and (content[i].isalnum() or content[i] in {".", "_"}):
                i += 1
            tokens.append(Token("number", content[start:i], start))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < len(content) and (content[i].isalnum() or content[i] == "_"):
                i += 1
            tokens.append(Token("ident", content[start:i], start))
            continue
        _raise_with_location(f"Unexpected character '{ch}'", path=path, content=content, offset=i)
    tokens.append(Token("eof", "", len(content)))
    return tokens


# =============================================================================
# AST
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Expr:
    offset: int


@dataclasses.dataclass(frozen=True)
class IdentExpr(Expr):
    name: str


@dataclasses.dataclass(frozen=True)
class StringExpr(Expr):
    value: str


@dataclasses.dataclass(frozen=True)
class NumberExpr(Expr):
    raw: str


@dataclasses.dataclass(frozen=True)
class VarExpr(Expr):
    name: str


@dataclasses.dataclass(frozen=True)
class SetRefExpr(Expr):
    name: str


@dataclasses.dataclass(frozen=True)
class ListExpr(Expr):
    items: List[Expr]


@dataclasses.dataclass(frozen=True)
class StructKey:
    offset: int


@dataclasses.dataclass(frozen=True)
class StringKey(StructKey):
    name: str


@dataclasses.dataclass(frozen=True)
class SetKey(StructKey):
    name: str


@dataclasses.dataclass(frozen=True)
class DefaultKey(StructKey):
    pass


@dataclasses.dataclass(frozen=True)
class StructExpr(Expr):
    entries: List[Tuple[StructKey, Expr]]


@dataclasses.dataclass(frozen=True)
class CallExpr(Expr):
    name: str
    args: List[Expr]


@dataclasses.dataclass(frozen=True)
class MemberExpr(Expr):
    base: Expr
    attr: str


@dataclasses.dataclass(frozen=True)
class AxisDecl:
    name: str
    expr: Expr
    offset: int


@dataclasses.dataclass
class BlockDef:
    name: str
    extends: str | None
    fields: Dict[str, Expr]
    path: Path
    source: str
    offset: int
    field_offsets: Dict[str, int]


@dataclasses.dataclass
class SetDef:
    name: str
    items: List[Expr]
    path: Path
    source: str
    offset: int


@dataclasses.dataclass
class KernelFamilyDef:
    name: str
    axes: List[AxisDecl]
    name_pattern: Expr
    with_expr: Expr | None
    fields: Dict[str, Expr]
    path: Path
    source: str
    offset: int
    field_offsets: Dict[str, int]


@dataclasses.dataclass
class Module:
    sets: Dict[str, SetDef] = dataclasses.field(default_factory=dict)
    fragments: Dict[str, BlockDef] = dataclasses.field(default_factory=dict)
    quant_traits: Dict[str, BlockDef] = dataclasses.field(default_factory=dict)
    quant_schemes: Dict[str, BlockDef] = dataclasses.field(default_factory=dict)
    families: List[KernelFamilyDef] = dataclasses.field(default_factory=list)
    operations: Dict[str, BlockDef] = dataclasses.field(default_factory=dict)
    categories: Dict[str, BlockDef] = dataclasses.field(default_factory=dict)


# =============================================================================
# Parser
# =============================================================================


class _Parser:
    def __init__(self, *, path: Path, content: str):
        self.path = path
        self.content = content
        self.tokens = _tokenize(path, content)
        self.idx = 0

    def _peek(self) -> Token:
        return self.tokens[self.idx]

    def _next(self) -> Token:
        tok = self.tokens[self.idx]
        self.idx += 1
        return tok

    def _expect(self, kind: str, value: str | None = None) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            _raise_with_location(f"Expected {kind}, got {tok.kind}", path=self.path, content=self.content, offset=tok.offset)
        if value is not None and tok.value != value:
            _raise_with_location(
                f"Expected '{value}', got '{tok.value}'", path=self.path, content=self.content, offset=tok.offset
            )
        return self._next()

    def _peek_ident_upper(self, value: str) -> bool:
        tok = self._peek()
        return tok.kind == "ident" and tok.value.upper() == value

    def _match_symbol(self, value: str) -> bool:
        tok = self._peek()
        if tok.kind == "symbol" and tok.value == value:
            self._next()
            return True
        return False

    def parse(self) -> Module:
        mod = Module()
        while True:
            tok = self._peek()
            if tok.kind == "eof":
                break
            if tok.kind != "ident":
                _raise_with_location("Expected declaration", path=self.path, content=self.content, offset=tok.offset)
            keyword = tok.value.upper()
            if keyword in {"DTYPE_SET", "SET"}:
                decl = self._parse_set()
                if decl.name in mod.sets:
                    _raise_with_location(
                        f"Duplicate set '{decl.name}'", path=self.path, content=self.content, offset=decl.offset
                    )
                mod.sets[decl.name] = decl
            elif keyword == "FRAGMENT":
                frag = self._parse_block("FRAGMENT")
                if frag.name in mod.fragments:
                    _raise_with_location(
                        f"Duplicate fragment '{frag.name}'", path=self.path, content=self.content, offset=frag.offset
                    )
                mod.fragments[frag.name] = frag
            elif keyword == "QUANT_SCHEME_TRAITS":
                traits = self._parse_block("QUANT_SCHEME_TRAITS")
                if traits.name in mod.quant_traits:
                    _raise_with_location(
                        f"Duplicate quant traits '{traits.name}'", path=self.path, content=self.content, offset=traits.offset
                    )
                mod.quant_traits[traits.name] = traits
            elif keyword == "QUANT_SCHEME":
                scheme = self._parse_block("QUANT_SCHEME")
                if scheme.name in mod.quant_schemes:
                    _raise_with_location(
                        f"Duplicate quant scheme '{scheme.name}'", path=self.path, content=self.content, offset=scheme.offset
                    )
                mod.quant_schemes[scheme.name] = scheme
            elif keyword == "OPERATION":
                op = self._parse_block("OPERATION", allow_bare_name=True)
                if op.name in mod.operations:
                    _raise_with_location(
                        f"Duplicate operation '{op.name}'", path=self.path, content=self.content, offset=op.offset
                    )
                mod.operations[op.name] = op
            elif keyword == "CATEGORY":
                cat = self._parse_block("CATEGORY", allow_bare_name=True)
                if cat.name in mod.categories:
                    _raise_with_location(
                        f"Duplicate category '{cat.name}'", path=self.path, content=self.content, offset=cat.offset
                    )
                mod.categories[cat.name] = cat
            elif keyword == "KERNEL_FAMILY":
                mod.families.append(self._parse_family())
            elif keyword == "INCLUDE":
                self._next()
                tok = self._peek()
                if tok.kind not in {"string", "ident"}:
                    _raise_with_location(
                        "INCLUDE expects a path string", path=self.path, content=self.content, offset=tok.offset
                    )
                self._next()
            else:
                _raise_with_location(
                    f"Unknown top-level declaration '{tok.value}'", path=self.path, content=self.content, offset=tok.offset
                )
        return mod

    def _parse_set(self) -> SetDef:
        start = self._expect("ident")
        name = start.value.upper()
        self._expect("symbol", "(")
        name_tok = self._expect("ident")
        self._expect("symbol", ")")
        self._expect("symbol", "{")
        items: List[Expr] = []
        while True:
            if self._match_symbol("}"):
                break
            tok = self._peek()
            if tok.kind == "symbol" and tok.value == "@":
                at = self._next()
                ref = self._expect("ident")
                items.append(SetRefExpr(offset=at.offset, name=ref.value))
            elif tok.kind == "ident":
                ident = self._next()
                items.append(IdentExpr(offset=ident.offset, name=ident.value))
            elif tok.kind == "string":
                s = self._next()
                items.append(StringExpr(offset=s.offset, value=s.value))
            else:
                _raise_with_location(
                    f"Invalid set item '{tok.value}'", path=self.path, content=self.content, offset=tok.offset
                )
            if self._match_symbol(","):
                continue
            self._expect("symbol", "}")
            break
        return SetDef(name=name_tok.value, items=items, path=self.path, source=self.content, offset=start.offset)

    def _parse_block(self, keyword: str, *, allow_bare_name: bool = False) -> BlockDef:
        start = self._expect("ident")
        if self._match_symbol("("):
            name_tok = self._expect("ident")
            self._expect("symbol", ")")
        elif allow_bare_name:
            name_tok = self._expect("ident")
        else:
            self._expect("symbol", "(")
            name_tok = self._expect("ident")
            self._expect("symbol", ")")
        extends: str | None = None
        if self._peek_ident_upper("EXTENDS"):
            self._next()
            parent = self._expect("ident")
            extends = parent.value
        self._expect("symbol", "{")
        fields, field_offsets = self._parse_field_list(allow_axis=False)
        self._expect("symbol", "}")
        return BlockDef(
            name=name_tok.value,
            extends=extends,
            fields=fields,
            path=self.path,
            source=self.content,
            offset=start.offset,
            field_offsets=field_offsets,
        )

    def _parse_family(self) -> KernelFamilyDef:
        start = self._expect("ident")
        self._expect("symbol", "(")
        name_tok = self._expect("ident")
        self._expect("symbol", ")")
        self._expect("symbol", "{")
        axes: List[AxisDecl] = []
        name_pattern: Expr | None = None
        with_expr: Expr | None = None
        fields: Dict[str, Expr] = {}
        field_offsets: Dict[str, int] = {}
        while True:
            if self._match_symbol("}"):
                break
            tok = self._peek()
            if tok.kind != "ident":
                _raise_with_location("Expected entry", path=self.path, content=self.content, offset=tok.offset)
            ident = self._next()
            ident_upper = ident.value.upper()
            if ident_upper == "AXIS":
                self._expect("symbol", "(")
                axis_name = self._expect("ident")
                self._expect("symbol", ")")
                self._expect("symbol", ":")
                expr = self._parse_expr()
                axes.append(AxisDecl(name=axis_name.value, expr=expr, offset=ident.offset))
            else:
                self._expect("symbol", ":")
                expr = self._parse_expr()
                if ident_upper == "NAME_PATTERN":
                    name_pattern = expr
                elif ident_upper == "WITH":
                    with_expr = expr
                else:
                    fields[ident_upper] = expr
                    field_offsets[ident_upper] = ident.offset
            if self._match_symbol(","):
                continue
            self._expect("symbol", "}")
            break
        if name_pattern is None:
            _raise_with_location(
                f"KERNEL_FAMILY({name_tok.value}) missing NAME_PATTERN",
                path=self.path,
                content=self.content,
                offset=start.offset,
            )
        return KernelFamilyDef(
            name=name_tok.value,
            axes=axes,
            name_pattern=name_pattern,
            with_expr=with_expr,
            fields=fields,
            path=self.path,
            source=self.content,
            offset=start.offset,
            field_offsets=field_offsets,
        )

    def _parse_field_list(self, *, allow_axis: bool) -> Tuple[Dict[str, Expr], Dict[str, int]]:
        fields: Dict[str, Expr] = {}
        offsets: Dict[str, int] = {}
        while True:
            if self._peek().kind == "symbol" and self._peek().value == "}":
                break
            tok = self._peek()
            if tok.kind != "ident":
                _raise_with_location("Expected field name", path=self.path, content=self.content, offset=tok.offset)
            name_tok = self._next()
            name_upper = name_tok.value.upper()
            if name_upper == "AXIS" and not allow_axis:
                _raise_with_location("AXIS not allowed here", path=self.path, content=self.content, offset=name_tok.offset)
            self._expect("symbol", ":")
            expr = self._parse_expr()
            fields[name_upper] = expr
            offsets[name_upper] = name_tok.offset
            if self._match_symbol(","):
                continue
            break
        return fields, offsets

    def _parse_expr(self) -> Expr:
        tok = self._peek()
        if tok.kind == "symbol" and tok.value == "[":
            return self._parse_list()
        if tok.kind == "symbol" and tok.value == "{":
            return self._parse_struct()
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        node = self._parse_atom()
        while self._match_symbol("."):
            attr = self._expect("ident")
            node = MemberExpr(offset=attr.offset, base=node, attr=attr.value)
        return node

    def _parse_atom(self) -> Expr:
        tok = self._peek()
        if tok.kind == "symbol" and tok.value == "@":
            at = self._next()
            name = self._expect("ident")
            return SetRefExpr(offset=at.offset, name=name.value)
        if tok.kind == "symbol" and tok.value == "$":
            at = self._next()
            name = self._expect("ident")
            return VarExpr(offset=at.offset, name=name.value)
        if tok.kind == "string":
            s = self._next()
            return StringExpr(offset=s.offset, value=s.value)
        if tok.kind == "number":
            n = self._next()
            return NumberExpr(offset=n.offset, raw=n.value)
        if tok.kind == "symbol" and tok.value == "*":
            s = self._next()
            return IdentExpr(offset=s.offset, name="*")
        if tok.kind == "ident":
            ident = self._next()
            if self._match_symbol("("):
                args: List[Expr] = []
                if not self._match_symbol(")"):
                    while True:
                        args.append(self._parse_expr())
                        if self._match_symbol(","):
                            continue
                        self._expect("symbol", ")")
                        break
                return CallExpr(offset=ident.offset, name=ident.value, args=args)
            return IdentExpr(offset=ident.offset, name=ident.value)
        _raise_with_location("Expected expression", path=self.path, content=self.content, offset=tok.offset)

    def _parse_list(self) -> Expr:
        start = self._expect("symbol", "[")
        items: List[Expr] = []
        if self._match_symbol("]"):
            return ListExpr(offset=start.offset, items=items)
        while True:
            items.append(self._parse_expr())
            if self._match_symbol(","):
                continue
            self._expect("symbol", "]")
            break
        return ListExpr(offset=start.offset, items=items)

    def _parse_struct(self) -> Expr:
        start = self._expect("symbol", "{")
        entries: List[Tuple[StructKey, Expr]] = []
        if self._match_symbol("}"):
            return StructExpr(offset=start.offset, entries=entries)
        while True:
            key_tok = self._peek()
            if key_tok.kind == "symbol" and key_tok.value == "*":
                t = self._next()
                key: StructKey = DefaultKey(offset=t.offset)
            elif key_tok.kind == "symbol" and key_tok.value == "@":
                t = self._next()
                name = self._expect("ident")
                key = SetKey(offset=t.offset, name=name.value)
            elif key_tok.kind == "ident":
                t = self._next()
                key = StringKey(offset=t.offset, name=t.value)
            elif key_tok.kind == "string":
                t = self._next()
                key = StringKey(offset=t.offset, name=t.value)
            else:
                _raise_with_location("Expected struct key", path=self.path, content=self.content, offset=key_tok.offset)
            self._expect("symbol", ":")
            value = self._parse_expr()
            entries.append((key, value))
            if self._match_symbol(","):
                continue
            self._expect("symbol", "}")
            break
        return StructExpr(offset=start.offset, entries=entries)


# =============================================================================
# Resolution / evaluation
# =============================================================================


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return ""

    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    suffix_multipliers = {"KB": 1024, "MB": 1024 * 1024, "GB": 1024 * 1024 * 1024}
    for suffix, multiplier in suffix_multipliers.items():
        if raw.endswith(suffix):
            number = raw[: -len(suffix)]
            if number.isdigit():
                return int(number) * multiplier

    if raw.replace(".", "", 1).isdigit():
        if "." in raw:
            return float(raw)
        return int(raw)

    return raw


@dataclasses.dataclass
class _ResolvedContext:
    path: Path
    source: str
    sets: Dict[str, List[Any]]
    fragments: Dict[str, "_ResolvedBlock"]
    quant_schemes: Dict[str, Dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class _ResolvedBlock:
    path: Path
    source: str
    fields: Dict[str, Expr]


def _expand_set(
    set_defs: Dict[str, SetDef],
    name: str,
    *,
    stack: List[str],
    path: Path,
    source: str,
) -> List[Any]:
    if name in stack:
        chain = " -> ".join(stack + [name])
        _raise_with_location(f"Cycle in set expansion: {chain}", path=path, content=source, offset=0)
    if name not in set_defs:
        _raise_with_location(f"Unknown set '{name}'", path=path, content=source, offset=0)
    stack.append(name)
    out: List[Any] = []
    for item in set_defs[name].items:
        if isinstance(item, SetRefExpr):
            out.extend(_expand_set(set_defs, item.name, stack=stack, path=path, source=source))
        elif isinstance(item, IdentExpr):
            out.append(item.name)
        elif isinstance(item, StringExpr):
            out.append(item.value)
        else:
            _raise_with_location("Unsupported set item", path=path, content=source, offset=item.offset)
    stack.pop()
    return out


_TEMPLATE_RE = re.compile(r"\{([^{}]+)\}")


def _format_placeholder(value: Any, spec: str) -> str:
    raw = "" if value is None else str(value)
    if spec == "":
        return raw
    if spec == "lower":
        return raw.lower()
    if spec == "upper":
        return raw.upper()
    if spec == "short":
        key = raw.upper()
        if key not in DTYPE_SHORT_MAP:
            raise KeyError(key)
        return DTYPE_SHORT_MAP[key]
    if spec.startswith("opt(") and spec.endswith(")"):
        prefix = spec[4:-1].strip()
        if len(prefix) >= 2 and prefix[0] == '"' and prefix[-1] == '"':
            prefix = prefix[1:-1]
        if raw == "":
            return ""
        return f"{prefix}{raw}"
    if spec.startswith("or(") and spec.endswith(")"):
        default = spec[3:-1].strip()
        if len(default) >= 2 and default[0] == '"' and default[-1] == '"':
            default = default[1:-1]
        return default if raw == "" else raw
    return raw


def _render_template(template: str, env: Dict[str, Any], *, path: Path, source: str, offset: int) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if ":" in inner:
            var, spec = inner.split(":", 1)
            var = var.strip()
            spec = spec.strip()
        else:
            var, spec = inner, ""
        if var not in env:
            _raise_with_location(f"Unknown template variable '{var}'", path=path, content=source, offset=offset)
        try:
            return _format_placeholder(env[var], spec)
        except KeyError:
            _raise_with_location(
                f"Unknown dtype for :short formatter: '{env[var]}'", path=path, content=source, offset=offset
            )
        return ""

    return _TEMPLATE_RE.sub(repl, template)


def _eval_expr(expr: Expr, env: Dict[str, Any], ctx: _ResolvedContext) -> Any:
    if isinstance(expr, IdentExpr):
        return expr.name
    if isinstance(expr, StringExpr):
        return _render_template(expr.value, env, path=ctx.path, source=ctx.source, offset=expr.offset)
    if isinstance(expr, NumberExpr):
        return _parse_scalar(expr.raw)
    if isinstance(expr, VarExpr):
        if expr.name not in env:
            _raise_with_location(f"Unknown variable '${expr.name}'", path=ctx.path, content=ctx.source, offset=expr.offset)
        return env[expr.name]
    if isinstance(expr, SetRefExpr):
        if expr.name not in ctx.sets:
            _raise_with_location(f"Unknown set '{expr.name}'", path=ctx.path, content=ctx.source, offset=expr.offset)
        return list(ctx.sets[expr.name])
    if isinstance(expr, ListExpr):
        items: List[Any] = []
        for item in expr.items:
            val = _eval_expr(item, env, ctx)
            if isinstance(val, list):
                items.extend(val)
            else:
                items.append(val)
        return items
    if isinstance(expr, StructExpr):
        has_special = any(isinstance(k, (SetKey, DefaultKey)) for k, _ in expr.entries)
        if has_special:
            _raise_with_location(
                "Mapping structs are only valid inside select(...)",
                path=ctx.path,
                content=ctx.source,
                offset=expr.offset,
            )
        out: Dict[str, Any] = {}
        for key, value_expr in expr.entries:
            if not isinstance(key, StringKey):
                _raise_with_location("Invalid struct key", path=ctx.path, content=ctx.source, offset=key.offset)
            out[key.name] = _eval_expr(value_expr, env, ctx)
        return out
    if isinstance(expr, CallExpr):
        name_upper = expr.name.upper()
        if name_upper == "SELECT":
            if len(expr.args) != 2:
                _raise_with_location("select(...) expects 2 arguments", path=ctx.path, content=ctx.source, offset=expr.offset)
            key_val = _eval_expr(expr.args[0], env, ctx)
            if not isinstance(expr.args[1], StructExpr):
                _raise_with_location("select(...) expects mapping struct as 2nd argument", path=ctx.path, content=ctx.source, offset=expr.offset)
            mapping = expr.args[1]
            default_expr: Expr | None = None
            for map_key, map_value in mapping.entries:
                if isinstance(map_key, DefaultKey):
                    default_expr = map_value
                    continue
                if isinstance(map_key, SetKey):
                    if map_key.name not in ctx.sets:
                        _raise_with_location(
                            f"Unknown set '{map_key.name}'",
                            path=ctx.path,
                            content=ctx.source,
                            offset=map_key.offset,
                        )
                    if key_val in ctx.sets[map_key.name]:
                        return _eval_expr(map_value, env, ctx)
                    continue
                if isinstance(map_key, StringKey):
                    if str(key_val) == map_key.name:
                        return _eval_expr(map_value, env, ctx)
                    continue
                _raise_with_location("Invalid select key", path=ctx.path, content=ctx.source, offset=map_key.offset)
            if default_expr is None:
                _raise_with_location("select(...) missing default '*'", path=ctx.path, content=ctx.source, offset=expr.offset)
            return _eval_expr(default_expr, env, ctx)
        if name_upper == "QUANT_SCHEME":
            if len(expr.args) != 1:
                _raise_with_location(
                    "QUANT_SCHEME(...) expects 1 argument", path=ctx.path, content=ctx.source, offset=expr.offset
                )
            scheme_name = _eval_expr(expr.args[0], env, ctx)
            if not isinstance(scheme_name, str):
                scheme_name = str(scheme_name)
            if scheme_name not in ctx.quant_schemes:
                _raise_with_location(
                    f"Unknown quant scheme '{scheme_name}'", path=ctx.path, content=ctx.source, offset=expr.offset
                )
            return ctx.quant_schemes[scheme_name]
        _raise_with_location(f"Unknown function '{expr.name}'", path=ctx.path, content=ctx.source, offset=expr.offset)
    if isinstance(expr, MemberExpr):
        base = _eval_expr(expr.base, env, ctx)
        if isinstance(base, dict):
            if expr.attr in base:
                return base[expr.attr]
            upper = expr.attr.upper()
            if upper in base:
                return base[upper]
        _raise_with_location(
            f"Unknown member '{expr.attr}'", path=ctx.path, content=ctx.source, offset=expr.offset
        )
    _raise_with_location("Unsupported expression", path=ctx.path, content=ctx.source, offset=expr.offset)


def _resolve_block_fields(
    defs: Dict[str, BlockDef],
    name: str,
    *,
    kind: str,
    stack: List[str],
) -> Dict[str, Expr]:
    if name in stack:
        chain = " -> ".join(stack + [name])
        base = defs[name]
        _raise_with_location(f"Cycle in {kind} inheritance: {chain}", path=base.path, content=base.source, offset=base.offset)
    if name not in defs:
        raise KeyError(name)
    stack.append(name)
    base = defs[name]
    merged: Dict[str, Expr] = {}
    if base.extends is not None:
        merged.update(_resolve_block_fields(defs, base.extends, kind=kind, stack=stack))
    merged.update(base.fields)
    stack.pop()
    return merged


def _resolve_quant_schemes(mod: Module, sets: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
    traits_resolved: Dict[str, Dict[str, Expr]] = {}
    for name in mod.quant_traits:
        traits_resolved[name] = _resolve_block_fields(mod.quant_traits, name, kind="quant traits", stack=[])

    schemes: Dict[str, Dict[str, Any]] = {}
    for name, scheme_def in mod.quant_schemes.items():
        merged_fields: Dict[str, Expr] = {}
        if scheme_def.extends is not None:
            if scheme_def.extends not in traits_resolved:
                _raise_with_location(
                    f"Unknown quant traits '{scheme_def.extends}'",
                    path=scheme_def.path,
                    content=scheme_def.source,
                    offset=scheme_def.offset,
                )
            merged_fields.update(traits_resolved[scheme_def.extends])
        merged_fields.update(scheme_def.fields)
        ctx = _ResolvedContext(path=scheme_def.path, source=scheme_def.source, sets=sets, fragments={}, quant_schemes={})
        evaluated: Dict[str, Any] = {}
        for key, expr in merged_fields.items():
            evaluated[key] = _eval_expr(expr, {}, ctx)
        schemes[name] = evaluated
    return schemes


def _resolve_fragments(mod: Module, sets: Dict[str, List[Any]]) -> Dict[str, _ResolvedBlock]:
    resolved: Dict[str, _ResolvedBlock] = {}
    for name, frag_def in mod.fragments.items():
        fields = _resolve_block_fields(mod.fragments, name, kind="fragment", stack=[])
        resolved[name] = _ResolvedBlock(path=frag_def.path, source=frag_def.source, fields=fields)
    _ = sets
    return resolved


def _expand_kernels(mod: Module, ctx: _ResolvedContext) -> List[KernelDescriptor]:
    kernels: List[KernelDescriptor] = []
    seen: Dict[str, KernelFamilyDef] = {}

    for family in mod.families:
        base_env: Dict[str, Any] = {}
        family_ctx = _ResolvedContext(
            path=family.path,
            source=family.source,
            sets=ctx.sets,
            fragments=ctx.fragments,
            quant_schemes=ctx.quant_schemes,
        )

        def expand_axis(idx: int, env: Dict[str, Any]) -> None:
            if idx >= len(family.axes):
                name_val = _eval_expr(family.name_pattern, env, family_ctx)
                if not isinstance(name_val, str):
                    name_val = str(name_val)
                if not re.fullmatch(r"[A-Za-z0-9_]+", name_val):
                    _raise_with_location(
                        f"Invalid kernel name '{name_val}'",
                        path=family.path,
                        content=family.source,
                        offset=family.offset,
                    )

                resolved_fields: Dict[str, Any] = {}
                for key, expr in family.fields.items():
                    resolved_fields[key] = _eval_expr(expr, env, family_ctx)

                if family.with_expr is not None:
                    with_val = _eval_expr(family.with_expr, env, family_ctx)
                    with_list = with_val if isinstance(with_val, list) else [with_val]
                    for frag_name_any in with_list:
                        frag_name = str(frag_name_any)
                        if frag_name not in family_ctx.fragments:
                            _raise_with_location(
                                f"Unknown fragment '{frag_name}'",
                                path=family.path,
                                content=family.source,
                                offset=family.offset,
                            )
                        frag = family_ctx.fragments[frag_name]
                        frag_ctx = _ResolvedContext(
                            path=frag.path,
                            source=frag.source,
                            sets=family_ctx.sets,
                            fragments=family_ctx.fragments,
                            quant_schemes=family_ctx.quant_schemes,
                        )
                        for key, expr in frag.fields.items():
                            resolved_fields[key] = _eval_expr(expr, env, frag_ctx)

                core_fields: Dict[str, Any] = {}
                extensions: Dict[str, Any] = {}
                for key, value in resolved_fields.items():
                    if key in CORE_FIELDS:
                        core_fields[key] = value
                    else:
                        extensions[key] = value

                if name_val in seen:
                    prev = seen[name_val]
                    _raise_with_location(
                        f"Duplicate kernel '{name_val}' (also generated by KERNEL_FAMILY({prev.name}))",
                        path=family.path,
                        content=family.source,
                        offset=family.offset,
                    )
                seen[name_val] = family

                locations = {"__kernel__": family.offset}
                locations.update(family.field_offsets)
                kernels.append(
                    KernelDescriptor(
                        name=name_val,
                        core_fields=core_fields,
                        extensions=extensions,
                        path=family.path,
                        source=family.source,
                        locations=locations,
                    )
                )
                return

            axis = family.axes[idx]
            axis_val = _eval_expr(axis.expr, env, ctx)
            values = axis_val if isinstance(axis_val, list) else [axis_val]
            for val in values:
                env2 = dict(env)
                env2[axis.name] = val
                expand_axis(idx + 1, env2)

        expand_axis(0, base_env)

    return kernels


def _iter_def_files(paths: Iterable[Path]) -> Iterator[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.def")))
        else:
            files.append(path)
    return iter(sorted(set(files)))


def parse_def_files(paths: Iterable[Path]) -> Module:
    merged = Module()
    for path in _iter_def_files(paths):
        content = strip_comments(path.read_text(encoding="utf-8"))
        mod = _Parser(path=path, content=content).parse()
        for name, s in mod.sets.items():
            if name in merged.sets:
                _raise_with_location(
                    f"Duplicate set '{name}'", path=path, content=content, offset=s.offset
                )
            merged.sets[name] = s
        for name, f in mod.fragments.items():
            if name in merged.fragments:
                _raise_with_location(
                    f"Duplicate fragment '{name}'", path=path, content=content, offset=f.offset
                )
            merged.fragments[name] = f
        for name, t in mod.quant_traits.items():
            if name in merged.quant_traits:
                _raise_with_location(
                    f"Duplicate quant traits '{name}'", path=path, content=content, offset=t.offset
                )
            merged.quant_traits[name] = t
        for name, q in mod.quant_schemes.items():
            if name in merged.quant_schemes:
                _raise_with_location(
                    f"Duplicate quant scheme '{name}'", path=path, content=content, offset=q.offset
                )
            merged.quant_schemes[name] = q
        for name, op in mod.operations.items():
            if name in merged.operations:
                _raise_with_location(
                    f"Duplicate operation '{name}'", path=path, content=content, offset=op.offset
                )
            merged.operations[name] = op
        for name, cat in mod.categories.items():
            if name in merged.categories:
                _raise_with_location(
                    f"Duplicate category '{name}'", path=path, content=content, offset=cat.offset
                )
            merged.categories[name] = cat
        merged.families.extend(mod.families)
    return merged


def collect_descriptors(paths: Iterable[Path]) -> List[KernelDescriptor]:
    mod = parse_def_files(paths)
    sets: Dict[str, List[Any]] = {}
    for name in mod.sets:
        sets[name] = _expand_set(mod.sets, name, stack=[], path=mod.sets[name].path, source=mod.sets[name].source)
    fragments = _resolve_fragments(mod, sets)
    schemes = _resolve_quant_schemes(mod, sets)
    ctx = _ResolvedContext(path=Path("<defs>"), source="", sets=sets, fragments=fragments, quant_schemes=schemes)
    return _expand_kernels(mod, ctx)


def validate_descriptor(desc: KernelDescriptor) -> None:
    required = ["OP", "INPUT_DTYPE", "OUTPUT_DTYPE", "ACCUM_DTYPE", "STRIDE_MODE"]
    for field in required:
        if field not in desc.core_fields:
            _raise_with_location(
                f"Kernel '{desc.name}' missing required field '{field}'",
                path=desc.path,
                content=desc.source,
                offset=desc.locations.get("__kernel__", 0),
            )

    op = str(desc.core_fields.get("OP", "")).lower()

    profiles = desc.core_fields.get("PROFILES")
    if profiles is not None:
        if not isinstance(profiles, dict):
            _raise_with_location(
                f"Kernel '{desc.name}' PROFILES must be a struct",
                path=desc.path,
                content=desc.source,
                offset=desc.locations.get("PROFILES", desc.locations.get("__kernel__", 0)),
            )
        normalized = {str(key).upper(): value for key, value in profiles.items()}
        if "SCALAR" not in normalized:
            _raise_with_location(
                f"Kernel '{desc.name}' PROFILES missing required SCALAR entry",
                path=desc.path,
                content=desc.source,
                offset=desc.locations.get("PROFILES", desc.locations.get("__kernel__", 0)),
            )
        for key, value in normalized.items():
            if not isinstance(value, str) or not value.strip():
                _raise_with_location(
                    f"Kernel '{desc.name}' PROFILES['{key}'] must be a non-empty string",
                    path=desc.path,
                    content=desc.source,
                    offset=desc.locations.get("PROFILES", desc.locations.get("__kernel__", 0)),
                )

    if "MATMUL_LAYOUT" not in desc.core_fields and op in {"matmul", "qkv_rope", "qkv_shared_input", "qkv_projection"}:
        _raise_with_location(
            f"Kernel '{desc.name}' missing MATMUL_LAYOUT for {op} operation",
            path=desc.path,
            content=desc.source,
            offset=desc.locations.get("__kernel__", 0),
        )

    if op in {"matmul", "qkv_rope", "qkv_shared_input", "qkv_projection"}:
        layout = str(desc.core_fields.get("MATMUL_LAYOUT", "INVALID")).upper()
        if layout not in {"NN", "NT", "TN", "TT"}:
            _raise_with_location(
                f"Kernel '{desc.name}' has invalid MATMUL_LAYOUT '{layout}' for {op} (expected NN|NT|TN|TT)",
                path=desc.path,
                content=desc.source,
                offset=desc.locations.get("MATMUL_LAYOUT", desc.locations.get("__kernel__", 0)),
            )
        if "WEIGHT_DTYPE" not in desc.core_fields and "WEIGHT_QUANT" not in desc.core_fields:
            _raise_with_location(
                f"Kernel '{desc.name}' for {op} requires WEIGHT_DTYPE or WEIGHT_QUANT",
                path=desc.path,
                content=desc.source,
                offset=desc.locations.get("__kernel__", 0),
            )

    if "WEIGHT_DTYPE" in desc.core_fields and "WEIGHT_QUANT" in desc.core_fields:
        _raise_with_location(
            f"Kernel '{desc.name}' has both WEIGHT_DTYPE and WEIGHT_QUANT (mutually exclusive)",
            path=desc.path,
            content=desc.source,
            offset=desc.locations.get("WEIGHT_QUANT", desc.locations.get("__kernel__", 0)),
        )

    if "WEIGHT_QUANT" in desc.core_fields:
        quant_block = desc.core_fields.get("QUANT_BLOCK")
        if quant_block is None:
            _raise_with_location(
                f"Kernel '{desc.name}' missing QUANT_BLOCK for quantized weights",
                path=desc.path,
                content=desc.source,
                offset=desc.locations.get("WEIGHT_QUANT", desc.locations.get("__kernel__", 0)),
            )
        required_subfields = {"block_size", "group_size", "scale_dtype", "zp_dtype"}
        if not isinstance(quant_block, dict):
            _raise_with_location(
                f"Kernel '{desc.name}' QUANT_BLOCK must be a struct",
                path=desc.path,
                content=desc.source,
                offset=desc.locations.get("QUANT_BLOCK", desc.locations.get("__kernel__", 0)),
            )
        missing = required_subfields - set(k.lower() for k in quant_block.keys())
        if missing:
            _raise_with_location(
                f"Kernel '{desc.name}' QUANT_BLOCK missing required subfields: {sorted(missing)}",
                path=desc.path,
                content=desc.source,
                offset=desc.locations.get("QUANT_BLOCK", desc.locations.get("__kernel__", 0)),
            )

    accum_dtype = str(desc.core_fields.get("ACCUM_DTYPE", "")).upper()
    if op in _ARITHMETIC_OPS and accum_dtype == "ANY":
        _raise_with_location(
            f"Kernel '{desc.name}' uses ACCUM_DTYPE ANY for arithmetic op '{op}'",
            path=desc.path,
            content=desc.source,
            offset=desc.locations.get("ACCUM_DTYPE", desc.locations.get("__kernel__", 0)),
        )

    if op in _LAYOUT_OPS and accum_dtype != "ANY":
        _raise_with_location(
            f"Kernel '{desc.name}' layout op '{op}' should use ACCUM_DTYPE ANY",
            path=desc.path,
            content=desc.source,
            offset=desc.locations.get("ACCUM_DTYPE", desc.locations.get("__kernel__", 0)),
        )

    if op in _REDUCTION_OPS and accum_dtype != "ANY":
        _raise_with_location(
            f"Kernel '{desc.name}' reduction op '{op}' should use ACCUM_DTYPE ANY",
            path=desc.path,
            content=desc.source,
            offset=desc.locations.get("ACCUM_DTYPE", desc.locations.get("__kernel__", 0)),
        )

    epilogue = desc.core_fields.get("EPILOGUE")
    activation = desc.core_fields.get("ACTIVATION")
    epilogue_list: List[str] = []
    if isinstance(epilogue, list):
        epilogue_list = [str(e).upper() for e in epilogue]
    elif epilogue is not None:
        epilogue_list = [str(epilogue).upper()]

    if "ACTIVATION" in epilogue_list and activation is None:
        _raise_with_location(
            f"Kernel '{desc.name}' EPILOGUE includes ACTIVATION but ACTIVATION field is missing",
            path=desc.path,
            content=desc.source,
            offset=desc.locations.get("EPILOGUE", desc.locations.get("__kernel__", 0)),
        )
    if activation is not None and "ACTIVATION" not in epilogue_list:
        _raise_with_location(
            f"Kernel '{desc.name}' has ACTIVATION field but EPILOGUE does not include ACTIVATION",
            path=desc.path,
            content=desc.source,
            offset=desc.locations.get("ACTIVATION", desc.locations.get("__kernel__", 0)),
        )


def validate_paths(paths: List[Path]) -> None:
    descriptors = collect_descriptors(paths)
    for desc in descriptors:
        validate_descriptor(desc)


# =============================================================================
# Quant scheme extraction (for Metal quantized matmul dispatch)
# =============================================================================


@dataclasses.dataclass
class QuantSchemeDescriptor:
    name: str
    impl_name: str
    uses_q8k_activation: bool
    use_direct_fp16_path: bool
    kernel_fp32: str | None
    kernel_fp32_out16: str | None
    kernel_fp16_out32: str | None
    kernel_fp16_out16: str | None
    qkv_kernel_fp32: str | None
    qkv_kernel_fp32_out16: str | None
    qkv_kernel_fp16_out32: str | None
    qkv_kernel_fp16_out16: str | None
    has_mv: bool = False
    has_mm: bool = False
    has_mm16: bool = False
    has_small: bool = False
    has_mv_ext: bool = False
    has_opt: bool = False
    has_opt_nr2: bool = False


def parse_quant_schemes(path: Path, content: str | None = None) -> List[QuantSchemeDescriptor]:
    if content is None:
        content = strip_comments(path.read_text(encoding="utf-8"))
    mod = _Parser(path=path, content=content).parse()
    sets: Dict[str, List[Any]] = {}
    for name in mod.sets:
        sets[name] = _expand_set(mod.sets, name, stack=[], path=mod.sets[name].path, source=mod.sets[name].source)
    schemes = _resolve_quant_schemes(mod, sets)
    out: List[QuantSchemeDescriptor] = []
    for name, fields in schemes.items():
        out.append(
            QuantSchemeDescriptor(
                name=name,
                impl_name=str(fields.get("IMPL_NAME", f"metal:quant:{name.lower()}")),
                uses_q8k_activation=bool(fields.get("USES_Q8K_ACTIVATION", False)),
                use_direct_fp16_path=bool(fields.get("USE_DIRECT_FP16_PATH", False)),
                kernel_fp32=fields.get("KERNEL_FP32"),
                kernel_fp32_out16=fields.get("KERNEL_FP32_OUT16"),
                kernel_fp16_out32=fields.get("KERNEL_FP16_OUT32"),
                kernel_fp16_out16=fields.get("KERNEL_FP16_OUT16"),
                qkv_kernel_fp32=fields.get("QKV_KERNEL_FP32"),
                qkv_kernel_fp32_out16=fields.get("QKV_KERNEL_FP32_OUT16"),
                qkv_kernel_fp16_out32=fields.get("QKV_KERNEL_FP16_OUT32"),
                qkv_kernel_fp16_out16=fields.get("QKV_KERNEL_FP16_OUT16"),
                has_mv=bool(fields.get("HAS_MV", False)),
                has_mm=bool(fields.get("HAS_MM", False)),
                has_mm16=bool(fields.get("HAS_MM16", False)),
                has_small=bool(fields.get("HAS_SMALL", False)),
                has_mv_ext=bool(fields.get("HAS_MV_EXT", False)),
                has_opt=bool(fields.get("HAS_OPT", False)),
                has_opt_nr2=bool(fields.get("HAS_OPT_NR2", False)),
            )
        )
    return out


def main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Parse and validate Marmot kernel .def descriptors")
    parser.add_argument("paths", nargs="+", type=Path, help="One or more .def files or directories containing .def files")
    args = parser.parse_args()
    try:
        validate_paths(args.paths)
    except DefParseError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    main()
