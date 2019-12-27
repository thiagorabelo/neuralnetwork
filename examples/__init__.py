
import argparse
import importlib
import sys


def main():
    def get_argv():
        argv = sys.argv
        if argv[0].endswith(__file__) \
                or argv[0].endswith('__main__.py')\
                or argv[0].endswith('__init__.py'):
            argv = argv[1:]
        return argv

    def open_module(module_name):
        try:
            module = importlib.import_module(f'examples.{module_name}')
        except ModuleNotFoundError as ex:
            raise argparse.ArgumentError(module_example_arg, str(ex))

        try:
            module.main
        except AttributeError as ex:
            raise argparse.ArgumentError(module_example_arg, str(ex))

        return module

    parser = argparse.ArgumentParser(description="Call some example module")
    module_example_arg = parser.add_argument('module_example',
                                             type=open_module,
                                             help="The module example that contains "
                                                  "a main function")
    options = parser.parse_args(get_argv())
    options.module_example.main()


if __name__ == "__main__":
    main()
