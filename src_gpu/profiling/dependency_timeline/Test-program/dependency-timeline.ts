type TokenKind =
  | "BEGIN_REF"
  | "END_REF"
  | "SPAN_OP"
  | "ARROW"
  | "DASH"
  | "ASSIGN_DEF"
  | "EQUAL"
  | "INTEGER"
  | "IDENT"
  | "UNIT"
  | "COMMENT"
  | "NEWLINE"
  | "EOF";

type Token = {
  kind: TokenKind;
  text: string;
  line: number;
  column: number;
};

type UnitName = "nanoseconds" | "microseconds" | "milliseconds" | "seconds";

type FileNode = {
  kind: "File";
  lines: LineNode[];
};

type LineNode =
  | { kind: "BlankLine" }
  | { kind: "CommentLine"; text: string }
  | { kind: "StatementLine"; statement: StatementNode; comment?: string };

type StatementNode = PipelineStmtNode | MetadataStmtNode;

type PipelineStmtNode = {
  kind: "PipelineStmt";
  segments: SegmentNode[];
};

type MetadataStmtNode = {
  kind: "MetadataStmt";
  interval: IntervalRefNode;
  timerName: string;
  duration: {
    value: number;
    unit: UnitName;
  };
};

type SegmentNode = SpanNode | IntervalRefNode;

type SpanNode = {
  kind: "Span";
  begin: BeginRefNode;
  end: EndRefNode;
  width: number;
};

type IntervalRefNode = {
  kind: "IntervalRef";
  begin: BeginRefNode;
  end: EndRefNode;
};

type BeginRefNode = {
  kind: "BeginRef";
  index: number;
  raw: string;
};

type EndRefNode = {
  kind: "EndRef";
  index: number;
  raw: string;
};

type ParseError = {
  kind: "ParseError";
  message: string;
  line: number;
  column: number;
};

type DyckError = {
  kind: "DyckError";
  message: string;
};

type Result =
  | { parseOk: true; dyckOk: true; ast: FileNode }
  | { parseOk: true; dyckOk: false; ast: FileNode; error: DyckError }
  | { parseOk: false; error: ParseError };

function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  let line = 1;
  let column = 1;

  function emit(kind: TokenKind, text: string, l = line, c = column) {
    tokens.push({ kind, text, line: l, column: c });
  }

  function advance(text: string) {
    for (const ch of text) {
      if (ch === "\n") {
        line += 1;
        column = 1;
      } else {
        column += 1;
      }
    }
    i += text.length;
  }

  while (i < input.length) {
    const rest = input.slice(i);

    const ws = /^[ \t]+/.exec(rest);
    if (ws) {
      advance(ws[0]);
      continue;
    }

    const newline = /^\r\n|\n/.exec(rest);
    if (newline) {
      emit("NEWLINE", newline[0]);
      advance(newline[0]);
      continue;
    }

    const comment = /^#[^\r\n]*/.exec(rest);
    if (comment) {
      emit("COMMENT", comment[0]);
      advance(comment[0]);
      continue;
    }

    const assignDef = /^:=/.exec(rest);
    if (assignDef) {
      emit("ASSIGN_DEF", assignDef[0]);
      advance(assignDef[0]);
      continue;
    }

    const arrow = /^->/.exec(rest);
    if (arrow) {
      emit("ARROW", arrow[0]);
      advance(arrow[0]);
      continue;
    }

    const spanOp = /^<\-+>/.exec(rest);
    if (spanOp) {
      emit("SPAN_OP", spanOp[0]);
      advance(spanOp[0]);
      continue;
    }

    const beginRef = /^begin_[0-9]+/.exec(rest);
    if (beginRef) {
      emit("BEGIN_REF", beginRef[0]);
      advance(beginRef[0]);
      continue;
    }

    const endRef = /^end_[0-9]+/.exec(rest);
    if (endRef) {
      emit("END_REF", endRef[0]);
      advance(endRef[0]);
      continue;
    }

    const unit = /^(nanoseconds|microseconds|milliseconds|seconds)\b/.exec(
      rest,
    );
    if (unit) {
      emit("UNIT", unit[1]);
      advance(unit[1]);
      continue;
    }

    const ident = /^[A-Za-z_][A-Za-z0-9_]*/.exec(rest);
    if (ident) {
      emit("IDENT", ident[0]);
      advance(ident[0]);
      continue;
    }

    const integer = /^[0-9]+/.exec(rest);
    if (integer) {
      emit("INTEGER", integer[0]);
      advance(integer[0]);
      continue;
    }

    const dash = /^-/.exec(rest);
    if (dash) {
      emit("DASH", dash[0]);
      advance(dash[0]);
      continue;
    }

    const equal = /^=/.exec(rest);
    if (equal) {
      emit("EQUAL", equal[0]);
      advance(equal[0]);
      continue;
    }

    throw {
      kind: "ParseError",
      message: `unexpected character: ${JSON.stringify(rest[0])}`,
      line,
      column,
    } satisfies ParseError;
  }

  tokens.push({ kind: "EOF", text: "", line, column });
  return tokens;
}

class Parser {
  private pos = 0;

  constructor(private tokens: Token[]) {}

  private peek(offset = 0): Token {
    return (
      this.tokens[this.pos + offset] ?? this.tokens[this.tokens.length - 1]
    );
  }

  private match(kind: TokenKind): boolean {
    return this.peek().kind === kind;
  }

  private consume(kind: TokenKind, message?: string): Token {
    const t = this.peek();
    if (t.kind !== kind) {
      throw {
        kind: "ParseError",
        message: message ?? `expected ${kind} but got ${t.kind}`,
        line: t.line,
        column: t.column,
      } satisfies ParseError;
    }
    this.pos += 1;
    return t;
  }

  parseFile(): FileNode {
    const lines: LineNode[] = [];
    while (!this.match("EOF")) {
      lines.push(this.parseLine());
    }
    this.consume("EOF");
    return { kind: "File", lines };
  }

  private parseLine(): LineNode {
    if (this.match("NEWLINE")) {
      this.consume("NEWLINE");
      return { kind: "BlankLine" };
    }

    if (this.match("COMMENT")) {
      const c = this.consume("COMMENT");
      if (this.match("NEWLINE")) this.consume("NEWLINE");
      return { kind: "CommentLine", text: c.text };
    }

    const statement = this.parseStatement();
    let comment: string | undefined;
    if (this.match("COMMENT")) {
      comment = this.consume("COMMENT").text;
    }
    if (this.match("NEWLINE")) {
      this.consume("NEWLINE");
    }
    return { kind: "StatementLine", statement, comment };
  }

  private parseStatement(): StatementNode {
    if (
      this.match("BEGIN_REF") &&
      this.peek(1).kind === "DASH" &&
      this.peek(2).kind === "END_REF" &&
      this.peek(3).kind === "ASSIGN_DEF"
    ) {
      return this.parseMetadataStmt();
    }
    return this.parsePipelineStmt();
  }

  private parsePipelineStmt(): PipelineStmtNode {
    const segments: SegmentNode[] = [];
    segments.push(this.parseSegment());
    while (this.match("ARROW")) {
      this.consume("ARROW");
      segments.push(this.parseSegment());
    }
    return { kind: "PipelineStmt", segments };
  }

  private parseMetadataStmt(): MetadataStmtNode {
    const interval = this.parseIntervalRef();
    this.consume("ASSIGN_DEF", "expected ':=' after interval reference");
    const timerName = this.consume("IDENT", "expected timer name").text;
    this.consume("EQUAL", "expected '=' after timer name");
    const valueTok = this.consume("INTEGER", "expected integer duration");
    const unitTok = this.consume("UNIT", "expected duration unit");

    return {
      kind: "MetadataStmt",
      interval,
      timerName,
      duration: {
        value: Number(valueTok.text),
        unit: unitTok.text as UnitName,
      },
    };
  }

  private parseSegment(): SegmentNode {
    const begin = this.parseBeginRef();
    if (this.match("SPAN_OP")) {
      const op = this.consume("SPAN_OP");
      const end = this.parseEndRef();
      return {
        kind: "Span",
        begin,
        end,
        width: op.text.length - 2,
      };
    }
    if (this.match("DASH")) {
      this.consume("DASH");
      const end = this.parseEndRef();
      return {
        kind: "IntervalRef",
        begin,
        end,
      };
    }
    const t = this.peek();
    throw {
      kind: "ParseError",
      message: "expected SPAN_OP or '-' after BEGIN_REF",
      line: t.line,
      column: t.column,
    } satisfies ParseError;
  }

  private parseIntervalRef(): IntervalRefNode {
    const begin = this.parseBeginRef();
    this.consume("DASH", "expected '-' in interval reference");
    const end = this.parseEndRef();
    return {
      kind: "IntervalRef",
      begin,
      end,
    };
  }

  private parseBeginRef(): BeginRefNode {
    const t = this.consume("BEGIN_REF", "expected begin_i");
    return {
      kind: "BeginRef",
      index: Number(t.text.slice("begin_".length)),
      raw: t.text,
    };
  }

  private parseEndRef(): EndRefNode {
    const t = this.consume("END_REF", "expected end_i");
    return {
      kind: "EndRef",
      index: Number(t.text.slice("end_".length)),
      raw: t.text,
    };
  }
}

function dyckCheck(ast: FileNode): DyckError | null {
  const stack: number[] = [];

  function visitSegment(seg: SegmentNode) {
    stack.push(seg.begin.index);
    const top = stack.pop();
    if (top !== seg.end.index) {
      throw {
        kind: "DyckError",
        message: `mismatched end: expected end_${top} but got end_${seg.end.index}`,
      } satisfies DyckError;
    }
  }

  function visitInterval(interval: IntervalRefNode) {
    stack.push(interval.begin.index);
    const top = stack.pop();
    if (top !== interval.end.index) {
      throw {
        kind: "DyckError",
        message: `mismatched end: expected end_${top} but got end_${interval.end.index}`,
      } satisfies DyckError;
    }
  }

  try {
    for (const line of ast.lines) {
      if (line.kind !== "StatementLine") continue;
      const st = line.statement;
      if (st.kind === "PipelineStmt") {
        for (const seg of st.segments) visitSegment(seg);
      } else {
        visitInterval(st.interval);
      }
    }
    if (stack.length !== 0) {
      return {
        kind: "DyckError",
        message: `unclosed begin_${stack[stack.length - 1]}`,
      };
    }
    return null;
  } catch (e) {
    return e as DyckError;
  }
}

export function parsePipelineTimes(input: string): Result {
  try {
    const tokens = tokenize(input);
    const parser = new Parser(tokens);
    const ast = parser.parseFile();
    const dyckError = dyckCheck(ast);
    if (dyckError) {
      return { parseOk: true, dyckOk: false, ast, error: dyckError };
    }
    return { parseOk: true, dyckOk: true, ast };
  } catch (e) {
    return { parseOk: false, error: e as ParseError };
  }
}
