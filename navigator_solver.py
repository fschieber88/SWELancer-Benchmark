# navigator_solver.py
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import chz
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import Step, FinalResult, FinalResultSuccessful
from nanoeval.solvers.computer_tasks.task import ComputerTask
from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface
from alcatraz.clusters.local import LocalConfig
import os
from pathlib import Path
import re
import asyncio

# ---------------------------------------------------------------------------
# Constants -----------------------------------------------------------------
# ---------------------------------------------------------------------------

# Decorate with chz to satisfy nanoeval requirements
@chz.chz
class NavigatorSolver(PythonCodingSolver):
    def shortname(self) -> str:              # shows up in the report
        return "navigator"

    @asynccontextmanager
    async def _start_computer(self, task: ComputerTask):
        # print(f"Starting computer for task: {task}")
        # Build a detailed TASK description for the container environment. We prefer
        # the dataset instructions if they are available on the `task.prompt` field.
        # Fallbacks: str(task) or the question ID if all else fails.
        try:
            prompt = getattr(task, "prompt", None)
            if prompt:
                task_description = _extract_issue_description(prompt)
            else:
                task_description = str(task)
        except Exception:
            task_description = str(getattr(task, "question_id", "SWELancer"))

        # ------------------------------------------------------------
        # Optional: enable Navigator's incremental git-aware scanner.
        # If the caller exported INCREMENTAL_SCAN=1 we inject the two
        # required environment variables *and* mount the cache volume
        # so that snapshots persist across benchmark containers.
        # ------------------------------------------------------------

        # Base environment that must always be present for the agent
        env = {
            "CODEBASE_PATH": "/app/expensify",
            "TASK": task_description,  # full issue
            "TASK_ID": str(getattr(task, "question_id", "1")),
            "NO_REDIS": "1",  # start immediately
            "DISPLAY": ":99",  # required for Playwright / Xvfb
        }

        # ------------------------------------------------------------------
        # Incremental-scanner support – always enabled inside the container.
        # ------------------------------------------------------------------

        env["INCREMENTAL_SCAN"] = "1"      # tell Navigator to use its cache
        env["SCANNER_CACHE_PATH"] = "/cache"  # fixed path expected by agent

        # Persist the incremental-scan cache --------------------------------
        host_cache = os.environ.get("SCANNER_CACHE_PATH")
        # Use a named Docker volume when the host path is not provided.
        cache_source = host_cache if host_cache else "nav-scan-cache"

        volumes_cfg = {
            "navigator_scan_cache": {
                "bind_source": cache_source,
                "bind_dest": "/cache",
                "mode": "rw",
            }
        }

        # Use navigator-agent image; its CMD already launches the agent, writes
        # the sentinel, and then sleeps forever to keep the container alive.
        cfg = task.model_copy(update={
            "docker_image": "navigator-agent:latest",
            "environment": env,
            "volumes_config": volumes_cfg,
        })
        # Determine which docker socket to use
        docker_host_env = os.environ.get("DOCKER_HOST")
        default_sock = "unix:///var/run/docker.sock"
        desktop_sock = f"unix://{os.path.expanduser('~/.docker/desktop/docker.sock')}"

        if docker_host_env:
            # Some Docker Desktop contexts set DOCKER_HOST to .../docker-cli.sock which
            # is *not* the REST API socket that docker-py expects.  Translate it to
            # the sibling docker.sock automatically.
            if docker_host_env.endswith("docker-cli.sock"):
                docker_host = docker_host_env.replace("docker-cli.sock", "docker.sock")
            else:
                docker_host = docker_host_env
        else:
            # Prefer the standard Linux socket if it exists.
            if os.path.exists(default_sock[len("unix://"):]):
                docker_host = default_sock
            # Fall back to Docker-Desktop socket *only* when the default one is absent.
            elif os.path.exists(desktop_sock[len("unix://"):]):
                docker_host = desktop_sock
            else:
                docker_host = default_sock  # let docker-py raise a clear error

        # Ensure docker-py and any subprocesses see the same socket
        os.environ["DOCKER_HOST"] = docker_host

        # Debug: show which docker socket/host we are using and whether it exists
        print("[NavigatorSolver] Using docker_host:", docker_host)
        if docker_host.startswith("unix://"):
            sock_path = docker_host[len("unix://"):]
            print("[NavigatorSolver] Socket path exists?", os.path.exists(sock_path))
        else:
            print("[NavigatorSolver] Non-unix DOCKER_HOST")

        async with task_to_alcatraz_config(
            cfg,
            LocalConfig(pull_from_registry=False, docker_host=docker_host),
        ).build() as cluster:
            # Yield control to the caller while the container is running
            try:
                yield AlcatrazComputerInterface(cluster_value=cluster)
            finally:
                # Always attempt to fetch container logs (even on crash) and write to file
                try:
                    # Retrieve a large tail to capture all logs. Docker returns the last N lines; using a very large
                    # number effectively fetches the complete logs.
                    logs: bytes = await cluster.fetch_container_logs(tail=100000)

                    # Ensure logs directory exists
                    logs_dir = Path("container_logs")
                    logs_dir.mkdir(exist_ok=True)

                    # Compose filename with task identifiers when available
                    try:
                        filename = f"{getattr(task, 'question_id', 'unknown')}_{getattr(task, 'attempt_id', '0')}_{getattr(task, 'retry_idx', 0)}.log"
                    except Exception:
                        filename = "navigator_container.log"

                    logfile_path = logs_dir / filename
                    logfile_path.write_bytes(logs)

                    print(f"[NavigatorSolver] Container logs written to {logfile_path}")
                except Exception as e:
                    # We purposefully swallow all exceptions here so that log collection never crashes the main flow
                    print(f"[NavigatorSolver] Failed to fetch/write container logs: {e}")

    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        async with self._start_computer(task) as comp:
            print(f"[NavigatorSolver] Starting agent")
            # Ensure required toolchain for SWELancer grading exists.
            # 1. sentinel so _setup proceeds.
            await comp.send_shell_command("echo done > /setup_done.txt")

            # Start a virtual X server so Playwright tests can launch headed Chromium.
            # We mimic the behaviour of the original /app/tests/run.sh which the base
            # image would have executed as ENTRYPOINT.
            await comp.send_shell_command("Xvfb :99 -screen 0 2560x1600x24 & disown")

            # 2. Node.js & npm (Ansible playbook expects npx).
            # Ensure the grader can call \`npx\`. The base swelancer image ships with Node via nvm
            # (\`/root/.nvm\`).  Newer Node versions (\u226520) no longer bundle a standalone npx
            # binary, so we create a tiny wrapper that loads nvm and delegates to \`npm exec\`.

            await comp.send_shell_command(
                "bash -c 'cat > /usr/bin/npx <<\"EOF\"\n#!/usr/bin/env bash\n# Lightweight replacement for the legacy npx binary.\n# We source nvm (available in base image) and then run npm exec so that\n# project-local binaries like webpack are found.\nif [ -f /root/.nvm/nvm.sh ]; then\n  source /root/.nvm/nvm.sh\n  # Activate the default node version so npm is on PATH\n  nvm use --silent >/dev/null 2>&1 || true\nfi\n# If npm still isn\'t on PATH, try common locations\nif ! command -v npm >/dev/null 2>&1; then\n  for candidate in /root/.nvm/versions/node/*/bin/npm; do\n    [ -x \"$candidate\" ] && export PATH=\"$(dirname \"$candidate\"):$PATH\" && break\n  done\nfi\nexec npm exec --yes \"$@\"\nEOF'"
            )
            await comp.send_shell_command("chmod +x /usr/bin/npx")
            await task.setup(comp)                       # puts repo under /root
            print(f"[NavigatorSolver] Agent started") # TK
            # Block until the agent signals completion by creating the sentinel file.
            print(f"[NavigatorSolver] Blocking until agent completes")

            # -----------------------------------------------------------------
            # Live log streaming – start `docker logs -f` in the background and
            # write everything to the same log file so we can inspect output
            # *while* the container is still running.
            # -----------------------------------------------------------------
            try:
                container_names = await comp.fetch_container_names()
                main_container_name = container_names[0] if container_names else None

                if main_container_name:
                    # Ensure logs directory and file path exist before spawning
                    logs_dir = Path("container_logs")
                    logs_dir.mkdir(exist_ok=True)
                    live_logfile_path = logs_dir / f"{getattr(task, 'question_id', 'unknown')}_{getattr(task, 'attempt_id', '0')}_{getattr(task, 'retry_idx', 0)}.live.log"

                    # Start `docker logs -f` as an async subprocess and pipe to file
                    log_proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "logs",
                        "-f",
                        main_container_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )

                    async def _pump_logs(proc, filepath):
                        with filepath.open("ab") as f:
                            assert proc.stdout is not None
                            async for chunk in proc.stdout:
                                f.write(chunk)
                                f.flush()

                    pump_task = asyncio.create_task(_pump_logs(log_proc, live_logfile_path))
                else:
                    log_proc = None
                    pump_task = None
            except Exception as e:
                print(f"[NavigatorSolver] Failed to start live log stream: {e}")
                log_proc = None
                pump_task = None

            exec_res = await comp.send_shell_command(
                f"while [ ! -f {SENTINEL_PATH} ]; do sleep 5; done"
            )
            print(f"[NavigatorSolver] Agent completed")
            # Append this exec output to the same log file we create in _start_computer
            try:
                logs_dir = Path("container_logs")
                logs_dir.mkdir(exist_ok=True)

                filename = f"{getattr(task, 'question_id', 'unknown')}_{getattr(task, 'attempt_id', '0')}_{getattr(task, 'retry_idx', 0)}.log"
                logfile_path = logs_dir / filename

                with logfile_path.open("ab") as f:
                    f.write(b"\n\n===== python /app/navigator/main.py output =====\n")
                    f.write(exec_res.output)
                    f.write(b"\n===== end =====\n")
            except Exception as e:
                print(f"[NavigatorSolver] Failed to append main.py output to log: {e}")

            # Snapshot all tasks (with context) and append to the same log so we
            # can inspect the hierarchy after the container exits.
            try:
                snapshot_cmd = "python /app/navigator-agent/task_cli.py list --all --context"
                snapshot_res = await comp.send_shell_command(snapshot_cmd)
                tasks_snapshot = snapshot_res.output

                with logfile_path.open("ab") as f:
                    f.write(b"\n===== tasks snapshot =====\n")
                    f.write(tasks_snapshot)
                    f.write(b"\n===== end =====\n")
            except Exception as e:
                print(f"[NavigatorSolver] Failed to record task snapshot: {e}")

            grade = await task.grade(comp)               # runs unit tests
            print(f"[NavigatorSolver] Grade: {grade}")
            yield FinalResultSuccessful(grade=grade)

            # Stop live log streaming if it was started.
            if log_proc:
                try:
                    log_proc.terminate()
                except Exception:
                    pass
            if pump_task:
                pump_task.cancel()

# ---------------------------------------------------------------------------
# Prompt-cleaning utility -----------------------------------------------------
# ---------------------------------------------------------------------------

_LEADING_META_RE = re.compile(
    r"^You are an .*?following issue:\s*", re.I | re.S
)

_TRAILING_SECTIONS_RE = re.compile(
    r"\n\n(?:Important Guidelines:|For your convenience,|When you are ready to submit|Important Guidance:).*$",
    re.I | re.S,
)

def _extract_issue_description(raw_prompt: "str | list") -> str:
    """Return only the GitHub issue block from a SWELancer prompt.

    The dataset stores prompts either as a list[dict(content=..)] or a raw
    string.  We drop the agent-instruction header and everything after the
    tool instructions to keep just the natural-language description of the
    task (Action performed, Expected, Actual, etc.).
    """

    # 1. Normalise to string with the first message's content if possible
    if isinstance(raw_prompt, list) and raw_prompt:
        raw_prompt = str(raw_prompt[0].get("content", ""))
    elif not isinstance(raw_prompt, str):
        raw_prompt = str(raw_prompt)

    txt: str = raw_prompt

    # 2. Remove leading "You are an expert … following issue:" preamble
    txt = _LEADING_META_RE.sub("", txt).lstrip()

    # 3. Cut off everything after the first tooling section
    txt = _TRAILING_SECTIONS_RE.sub("", txt).rstrip()

    return txt

# Path of the Navigator entry-point script inside the Docker image.
AGENT_PATH = "/app/navigator/main.py"
# File that the agent touches when it has completed all work.
SENTINEL_PATH = "/app/navigator/FINISHED"