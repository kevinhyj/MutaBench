# Single-chain specialist adapters

Project-owned adapters are under `singlechain/`, with `run.py` as the original
entrypoint. Model weights, third-party source and environments are excluded.
The reference code retains its original input contracts and external-tool
requirements; it does not automatically consume the new MutaBench CSV interface.

Use `run.py --help` to inspect the original contract after installing its
dependencies. Configure model assets and input paths explicitly. Adapter source
presence does not mean all external models are installed, available or applicable
to every benchmark task. Historical details are in `../catalog.json`.

