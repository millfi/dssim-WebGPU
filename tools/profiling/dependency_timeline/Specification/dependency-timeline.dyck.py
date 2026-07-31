import re

from lark import Lark

grammar = r"""
start: line*

?line: statement _NL           -> statement_line
     | COMMENT _NL             -> comment_line
     | _NL                     -> blank_line
     | statement COMMENT?      -> eof_statement_line
     | COMMENT                 -> eof_comment_line

?statement: pipeline_stmt
          | metadata_stmt

pipeline_stmt: segment (ARROW segment)*
metadata_stmt: interval_ref ASSIGN NAME EQ INT UNIT

?segment: span
        | interval_ref

span: begin_ref SPANOP end_ref
interval_ref: begin_ref DASH end_ref
begin_ref: BEGIN INT
end_ref: END INT

ARROW: "->"
ASSIGN: ":="
DASH: "-"
BEGIN: "begin_"
END: "end_"
EQ: "="
UNIT: "nanoseconds" | "microseconds" | "milliseconds" | "seconds"
NAME: /[A-Za-z_][A-Za-z0-9_]*/
SPANOP: /<[-]+>/
COMMENT: /#[^\n\r]*/
_NL: /\r?\n/

%import common.INT
%import common.WS_INLINE
%ignore WS_INLINE
"""

parser = Lark(grammar, parser="lalr")

sample_ok = """# 1行コメント
begin_0 <-->end_0 -> begin_1 <------->end_1 -> begin_2 <->end_2 # 「<->」の-の個数は自由, 「A -> B」はBはAが終わらないと始めない依存関係を表す
begin_0-end_0 -> begin_3 <-->end_3 -> begin_4 <-->end_4 # 非同期処理による1行前からのpiplineの分離の表現
begin_3-end_3 -> begin_5 <--> end_5 # 同様にまた分離できる
begin_3-end_3 := timer_name_3 = 10 microseconds # 処理区間 = プロファイリングタイマーの名前 = その計測時間
begin_1-end_1 := timer_name_1 = 2 seconds # 簡単のため重複は許可、使用する単位はnanoseconds, microseconds, milliseconds, secondsの4つ。小数は簡単のため許可しない
"""

print("sample_ok parse ok?", bool(parser.parse(sample_ok)))


def dyck_check(text):
    no_comments = re.sub(r"#[^\n\r]*", "", text)
    toks = re.findall(r"begin_(\d+)|end_(\d+)", no_comments)
    seq = []
    for b, e in toks:
        if b:
            seq.append(("b", int(b)))
        else:
            seq.append(("e", int(e)))
    stack = []
    for typ, i in seq:
        if typ == "b":
            stack.append(i)
        else:
            if not stack or stack[-1] != i:
                return False, seq
            stack.pop()
    return not stack, seq
