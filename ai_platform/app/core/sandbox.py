import docker
import tempfile
from pathlib import Path
import shutil
import asyncio

class SecureDockerExecutor:
    def __init__(
        self,
        image: str = "python:3.11-alpine",
        memory_limit: str = "128m",
        cpu_quota: int = 50000,
        cpu_period: int = 100000,
        timeout: int = 10,
        max_output_size: int = 1024 * 1024
    ):
        self.client = docker.from_env()
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.cpu_period = cpu_period
        self.timeout = timeout
        self.max_output_size = max_output_size

        try:
            self.client.images.get(image)
        except docker.errors.ImageNotFound:
            self.client.images.pull(image)

    def run_python(self, code: str) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "main.py"
            script_path.write_text(code, encoding='utf-8')

            dockerfile_content = f"""
FROM {self.image}
RUN adduser -D -u 1000 sandbox
WORKDIR /sandbox
COPY main.py .
RUN chown -R sandbox:sandbox /sandbox
USER sandbox
CMD ["python", "main.py"]
"""
            (Path(tmpdir) / "Dockerfile").write_text(dockerfile_content)

            try:
                image, _ = self.client.images.build(
                    path=tmpdir,
                    dockerfile="Dockerfile",
                    nocache=True,
                    rm=True
                )

                container = self.client.containers.run(
                    image.id,
                    detach=True,
                    network_mode="none",
                    mem_limit=self.memory_limit,
                    cpu_quota=self.cpu_quota,
                    cpu_period=self.cpu_period,
                    pids_limit=50,
                    read_only=True,
                    tmpfs={'/tmp': 'rw,noexec,nosuid,size=10m'},
                    remove=False
                )

                try:
                    result = container.wait(timeout=self.timeout)
                    logs = container.logs(stdout=True, stderr=True, tail=1000)

                    if len(logs) > self.max_output_size:
                        logs = logs[:self.max_output_size] + b"\n...truncated"

                    return {
                        "success": result["StatusCode"] == 0,
                        "returncode": result["StatusCode"],
                        "output": logs.decode('utf-8', errors='replace')
                    }
                finally:
                    container.remove(force=True)
                    self.client.images.remove(image.id, force=True)

            except docker.errors.ContainerError as e:
                return {
                    "success": False,
                    "returncode": e.exit_status,
                    "output": e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
                }
            except Exception as e:
                return {
                    "success": False,
                    "returncode": -1,
                    "output": str(e)
                }
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    async def run_python_async(self, code: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run_python, code)
