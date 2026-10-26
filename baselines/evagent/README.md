# EVAgent reference implementation

This is the maintained EVAgent package exported from the research project.
It includes CR/ACR, evidence, few-shot and research protocol implementations.
Historical modules and model identifiers retain their original names and
behavior. They are reference methods within MutaBench, rather than the benchmark
submission format.

Install in a separate environment if you need this reference method:

```bash
python -m pip install -e baselines/evagent
python -m evagent.main experiment \
  --config baselines/evagent/examples/cr_direct.json --dry-run
```

The example inputs are synthetic. The command above is a no-call configuration
check. Live model generation requires your own explicit provider configuration,
credentials and selected inputs. Use new output directories. Provider defaults
in legacy source can include historical model names and local proxy paths;
configure them for your host instead of assuming those services exist.

To run the exported tests without network access, install pytest and the package
dependencies, then run:

```bash
cd baselines/evagent
python tools/offline_pytest.py -q
```

Optional agent/scientific backends are separate dependencies. Five tests that
require original structure files or reviewed literature passages are explicitly
skipped when those separate assets are absent. Bibliographic metadata and
paraphrased guidance are included; raw structure/literature snapshots, installed
tools, private task files and host-specific tool wrappers are not bundled.

The exported seed-profile JSON files are historical test fixtures, including
their loopback proxy and old configured timeout. For a new live run, provide
your own configuration with an explicit endpoint/proxy and a generation wait
of at least 3,600 seconds. Historical fixtures are not recommended live defaults.
