# Sandbox security model

TradeBench v0 has two sandbox providers:

- `docker`: the real benchmark provider
- `local`: a convenience provider for unit tests only

The benchmark only treats the Docker provider as part of the security story.

## What the Docker provider does guarantee

`tradebench.sandbox.providers.docker.DockerSandboxRunner` creates containers with
these properties:

- container networking disabled via `network_disabled=True`
- Docker proxy environment injection disabled via `use_config_proxy=False`
- read-only container root filesystem
- non-root process user `1000:1000`
- `no-new-privileges:true`
- host seccomp profile mounted through Docker security options
- writable binds only for `/workspace/work` and `/workspace/out`
- read-only binds for `/workspace/data` and `/workspace/meta`
- no automatic host home-directory mount
- no automatic merge of host `os.environ`

Inside the container, the default environment is intentionally tiny:

- `PYTHONUNBUFFERED=1`
- `PYTHONNOUSERSITE=1`
- `HOME=/home/sandbox`

Any additional environment variables must be supplied explicitly by the caller.

## What the Docker provider does not guarantee

TradeBench v0 is a hardened container workflow, not a VM or a formally verified
isolation boundary. It does not promise:

- protection against Docker, kernel, or container-escape vulnerabilities
- CPU, memory, PID, or disk quotas
- secret scrubbing from the sandbox image itself
- deterministic filesystem contents beyond the image tag the operator builds
- protection from a privileged host user or a compromised Docker daemon
- safe execution of arbitrary code outside the benchmark workflow

The environment relies on the ledger and tool surface for benchmark integrity, not
on the sandbox for correctness.

## What the local provider means

`tradebench.sandbox.providers.local.LocalSandboxRunner` runs commands directly on
the host with `subprocess.run(...)`.

It does not guarantee:

- filesystem isolation
- environment isolation
- network isolation
- protection from host-side side effects

Use it only for unit tests and other non-security-sensitive checks.

## Benchmark integrity vs sandbox isolation

Even when the sandbox is working correctly, the sandbox is not the authority for:

- cash balances
- positions
- open orders
- fills
- reward
- score

Those live in the environment runtime and are derived from ledger events. A
sandboxed process can write files in the episode workspace, but it cannot mutate
portfolio state except by calling environment tools that emit validated events.

## Practical release-gate expectation

A fresh clone should be able to:

1. build `docker/sandbox.Dockerfile` into the default image tag used by the CLI
2. start a session with `sandbox_provider="docker"`
3. run the sample split without editing benchmark data by hand

That is the operational claim v0 makes. It is intentionally narrower than a claim
of complete host isolation.
