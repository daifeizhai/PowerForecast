"""Numerically compare the transferred solver with the user's original source."""
import argparse
import ast
from hashlib import sha256
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.models.proposed.simplex import fit_simplex
from src.utils import write_json


def audit(path):
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in {"weighted_breakpoint", "fit_simplex"}]
    if len(functions) != 2:
        raise ValueError("Expected the two original simplex functions.")
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    rows = []
    for members in (4, 5, 8, 9):
        rng = np.random.default_rng(100+members)
        P = rng.normal(1000., 120., size=(1400, members))
        y = P@rng.dirichlet(np.ones(members))+rng.normal(0., 20., len(P))
        valid = rng.random(len(P)) > .02
        expected = namespace["fit_simplex"](P, y, valid)
        actual = fit_simplex(P, y, valid)
        np.testing.assert_array_equal(expected, actual)
        rows.append(dict(members=members, points=len(y), max_weight_difference=float(np.max(np.abs(expected-actual)))))
    output = dict(reference=str(Path(path).resolve()), reference_sha256=sha256(source.encode()).hexdigest(),
                  optimizer_identical=True, checks=rows)
    write_json("results/analysis/nlr/core-fidelity.json", output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to old power_forecast/attainment_fusion.py")
    audit(parser.parse_args().source)
