from __future__ import print_function, unicode_literals


def main():
    import sys

    import plac


    commands = {
        "train": train
    }
    if len(sys.argv) == 1:
        pretty_print(', '.join(commands), title="Available commands", exits=1,
                     level=PrettyPrintLevel.INFO)
    command = sys.argv.pop(1)
    sys.argv[0] = 'projection-net %s' % command
    if command in commands:
        plac.call(commands[command], sys.argv[1:])
    else:
        pretty_print("Available: %s" % ', '.join(commands),
                     title="Unknown command: %s" % command, exits=1,
                     level=PrettyPrintLevel.INFO)


if __name__ == "__main__":
    main()