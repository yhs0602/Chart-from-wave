import readline
import sqlite3
from functools import partial

import subprocess


def write_to_clipboard(output):
    process = subprocess.Popen(
        "pbcopy", env={"LANG": "en_US.UTF-8"}, stdin=subprocess.PIPE
    )
    process.communicate(output.encode("utf-8"))


def completer(text, state, completion_options):
    # Get all completion options that start with the current text
    options = [option for option in completion_options if option.startswith(text)]
    if state < len(options):
        return options[state]
    else:
        return None


def input_with_tab_completion(prompt, options):
    # Set the completer function for readline
    readline.set_completer(partial(completer, completion_options=options))
    if "libedit" in readline.__doc__:
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

    # Call input with the provided prompt
    return input(prompt)


def main():
    filename = input("Enter file path")
    con = sqlite3.connect(filename)

    cur = con.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    table_names = [table[0] for table in tables]
    print("Tables in the database:", table_names)

    table_name = input_with_tab_completion("Enter table name", table_names)
    print("Table name:", table_name)

    cur.execute(f"PRAGMA table_info({table_name});")
    columns = cur.fetchall()

    column_names = [column[1] for column in columns]
    print("Columns in the table:", column_names)
    column_name = input_with_tab_completion("Enter column name to list", column_names)
    print("Column name:", column_name)

    cur.execute(f"SELECT {column_name} FROM {table_name};")
    column_values = cur.fetchall()

    print(f"Column values ({len(column_values)}):")
    for value in column_values:
        print(value[0])

    column_values = [value[0] for value in column_values]
    column_value = input_with_tab_completion(
        "Enter column value to list", column_values
    )
    print("Column value:", column_value)
    column_to_read = input_with_tab_completion("Enter column to read", column_names)
    print("Column to read:", column_to_read)

    cur.execute(
        f"SELECT {column_to_read} FROM {table_name} WHERE {column_name} = '{column_value}';"
    )
    column_values = cur.fetchall()
    value = column_values[0][0]
    str_value = value.decode("utf-8")
    write_to_clipboard(str_value)
    print(f"Value: {str_value} (copied to clipboard)")


if __name__ == "__main__":
    main()
