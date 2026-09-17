"""Argument parser with stdout diagnostics for the project's standalone scripts."""

import argparse
import sys


class ScriptParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stdout)
        print(f"{self.prog}: {message}")
        self.exit(2)
