# σ-calculus

[crates/why-data/src/graph/sigmacalculus.rs](../../crates/why-data/src/graph/sigmacalculus.rs)
ports the book's `SigmaCalculusInspector` (Correa & Bareinboim 2020), reusing
`docalculus`'s formatting helpers. It deliberately departs from the Python in
three places, listed in its module docs. The Python has a mutable-default-list
bug in `TransformNodes` that leaks cuts between queries. So when its answers
disagree with this module, check the Python first. Its test graphs were
cross-checked by running the Python inspector, which needs `networkx`,
`toposort` and `pydash`. There is no σ-calculus notebook, so there is no
`examples/` port. `why_parser::fusion` reads the book's `<QUERY>` and
`<INTERVENTIONS>` sections, and `Document::sigma_calculus` runs the rule they
describe. Watch the naming: `treatment:` is X in `<TASK>` but Z in `<QUERY>`.
