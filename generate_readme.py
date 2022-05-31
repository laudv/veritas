import os, re, io
from contextlib import redirect_stdout

CODE_FILE = "tests/readme_code.py"
TEMPLATE_FILE = "README_TEMPLATE.md"
TARGET_FILE = "README.md"

def __parse_code_parts(f):
    parts = {}
    active_part = None
    active_part_line = 0
    for i, line in enumerate(f.readlines()):
        m0 = re.search("^### <PART ?([a-z0-9_-]+)>[ \t]*$", line, re.IGNORECASE)
        m1 = re.search("^### </PART>[ \t]*$", line, re.IGNORECASE)
        if m0:
            #print(f"{i:04}: open", m0.group(1))
            active_part = m0.group(1)
            active_part_line = i+1
            parts[active_part] = io.StringIO()
        elif m1:
            print(f"{i:04}: {active_part} lines {active_part_line}:{i}")
            active_part = None
        elif active_part:
            parts[active_part].write(line)
    return {k: v.getvalue() for k, v in parts.items()}

def __check_in_parts(parts, part, line, file):
    if part not in parts:
        raise RuntimeError(f"part with name `{part}` not defined ({file}:{line})")

def __parse_insertion_parts(f, parts):
    globals = {}
    output = io.StringIO()

    for i, line in enumerate(f.readlines()):
        m0 = re.search("^!code PART ?([a-z0-9_-]+)![ \t]*$", line, re.IGNORECASE)
        m1 = re.search("^!output PART ?([a-z0-9_-]+)( LINES ([0-9]+):([0-9]+))?![ \t]*$",
                line, re.IGNORECASE)
        if m0: # insert python code part
            part = m0.group(1)
            __check_in_parts(parts, part, i, TEMPLATE_FILE)
            print(f"{i:04}: inserting code", part)
            print("```python", file=output)
            output.write(parts[part])
            print("```", file=output)
        elif m1: # insert python output of code part
            part = m1.group(1)
            __check_in_parts(parts, part, i, TEMPLATE_FILE)
            print(f"{i:04}: inserting output", part)
            print("```", file=output)
            exec_out = __execute_part(parts[part], globals)
            line0, line1 = m1.group(3, 4)
            if line0 is not None and line1 is not None:
                try:
                    line0, line1 = int(line0), int(line1)
                except ParseError as e:
                    print("invalid format", line)
                print("   ", f"selecting lines {line0}:{line1} FROM")
                print("\n".join(map(lambda x: f"    {x[0]:<3d}| {x[1]}", enumerate(exec_out.splitlines()))))
                exec_out = "".join(list(exec_out.splitlines(keepends=True))[line0:line1])
                print("   ", "TO")
                print("\n".join(map(lambda x: f"    {x[0]:<3d}| {x[1]}", enumerate(exec_out.splitlines()))))
            output.write(exec_out)
            print("```", file=output)
        else:
            output.write(line) # just copy line
    return output

def __execute_part(code, globals):
    f = io.StringIO()
    with redirect_stdout(f):
        exec(code, globals)
    return f.getvalue()

if __name__ == "__main__":
    with open(CODE_FILE) as f:
        parts = __parse_code_parts(f)
    print()
    with open(TEMPLATE_FILE) as f:
        output = __parse_insertion_parts(f, parts)
    print()
    with open(TARGET_FILE, "w") as f:
        f.write(output.getvalue())

    print("readme written", TARGET_FILE)
