import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import paramiko
from rich.progress import Progress
from scp import SCPClient

from hermes.cloudbreak.clouds.base.resource import Resource
from hermes.cloudbreak.logging import logger

Response = Optional[str]
Responses = List[Response]


# TODO: have this inherit from ManagerResource
class VMManager:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self._resources = []

    @property
    def resources(self):
        return {vm.name: vm for vm in self._resources}

    @property
    def N(self):
        return len(self._resources)

    @property
    def client(self):
        return self.parent.client

    def create_one_vm(self, name):
        raise NotImplementedError

    @contextmanager
    def manage(self, N: int, username: str, ssh_key_file: str):
        self.create(N, username, ssh_key_file)
        self.wait_for_ready()

        try:
            yield self
        finally:
            self.delete()
            self.wait_for_delete()

    def delete(self):
        for vm in self._resources:
            vm.delete()

    def wait_for_delete(self):
        with Progress() as progbar:
            task_id = progbar.add_task(
                "Waiting for VMs to be delete", total=self.N
            )
            while self._resources:
                completed = [vm for vm in self._resources if vm.is_deleted()]
                for vm in completed:
                    self._resources.remove(vm)

                progbar.update(task_id, advance=len(completed))

    def create(self, N: int, username: str, ssh_key_file: str):
        for i in range(N):
            name = f"{self.name}-{i}"
            try:
                vm = self.resources[name]
            except KeyError:
                vm = self.create_one_vm(name)
                self._resources.append(vm)

            vm.set_user(username, ssh_key_file)

    def wait_for_ready(self):
        with Progress() as progbar:
            task_id = progbar.add_task(
                "Waiting for VMs to be ready", total=self.N
            )
            while not progbar.finished:
                n = 0
                for vm in self._resources:
                    if vm.is_ready():
                        n += 1
                progbar.update(task_id, completed=n)

    def run(self, cmd, *kwargs):
        # TODO: what's a more intelligent way of handling this?
        n_jobs = min(self.N, 32)
        futures = []
        with Progress() as progbar, ThreadPoolExecutor(n_jobs) as ex:
            submit_task_id = progbar.add_task(
                "Submitting commands", total=self.N
            )
            done_task_id = progbar.add_task(
                "Waiting for responses", total=self.N
            )

            # submit the run command on each one of the vms
            for vm in self._resources:
                futures.append(ex.submit(vm.run, cmd))

            stdouts, stderrs, exceptions = {}, {}, {}
            while not progbar.finished:
                # if not all the jobs are running yet, update
                # the submit progress bar with the ones that are
                if not progbar._tasks[submit_task_id].finished:
                    n_submitted = len(
                        [f for f in futures if f.running() or f.done()]
                    )
                    progbar.update(submit_task_id, completed=n_submitted)

                # now iterate through all of the submitted
                # futures and check for completion/errors
                for vm, f in zip(self._resources, futures):
                    if vm.name in stdouts or vm.name in exceptions:
                        # we've already dealt with this future
                        continue

                    try:
                        exc = f.exception(0.001)
                    except TimeoutError:
                        # this job hasn't finished yet, so move on
                        continue
                    else:
                        if exc is not None:
                            # the job finished but it raised an exception
                            exceptions.append(exc)
                        else:
                            # the job finished and there was no exception
                            # so grab the result from the finished future
                            stdout, stderr = f.result()
                            stdouts[vm.name] = stdout
                            stderrs[vm.name] = stderr

                        # either way, we have a job that's done, so
                        # update our progress bar to reflect that
                        progbar.update(done_task_id, advance=1)

            # log all the exceptions we ran into but
            # just raise the last of them
            if len(exceptions) > 0:
                for name, exc in exceptions.items():
                    logger.error(f"Encountered error {exc} on VM {name}")
                raise exc

        return stdouts, stderrs


@dataclass
class VMInstance(Resource):
    def __post_init__(self):
        self._ip = None
        self._start_time = time.time()
        self._is_ready = False

        self.username = None
        self.ssh_key_file = None

    @property
    def ip(self):
        if self._ip is not None:
            return self._ip

    @property
    def message(self):
        return f"VM {self.name}"

    def set_user(self, username: str, ssh_key_file: str):
        self.username = username
        self.ssh_key_file = ssh_key_file

    @contextmanager
    def connect(
        self,
        scp: bool = False,
        username: Optional[str] = None,
        ssh_key_file: Optional[str] = None,
    ):
        username = username or self.username
        ssh_key_file = ssh_key_file or self.ssh_key_file

        if username is None:
            raise ValueError("Must specify username")
        elif ssh_key_file is None:
            raise ValueError("Must specify ssh key file")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())

        # try to push through possible connection issues,
        # but log them for the sake of debugging
        for i in range(5):
            try:
                client.connect(
                    username=username,
                    hostname=self.ip,
                    key_filename=ssh_key_file,
                )
                break
            except Exception as e:
                logger.warning(
                    "Failed to connect to VM {} at IP {} on attempt {} "
                    "after receiving error {}. Trying again".format(
                        self.name, self.ip, i + 1, str(e)
                    )
                )
                time.sleep(1)
        else:
            # if after 5 attempts we were unable to
            # connect to the instance, raise an error
            raise RuntimeError(
                "Failed to connect to VM {} at IP {} after 5 attempts".format(
                    self.name, self.ip
                )
            )

        # create an scp client if needed
        if scp:
            ssh_client = client
            client = SCPClient(ssh_client.get_transport())

        # return the client then close everything
        # at the end, even if something goes wrong
        try:
            yield client
        finally:
            client.close()
            if scp:
                ssh_client.close()

    def run(
        self,
        *cmds,
        username: Optional[str] = None,
        ssh_key_file: Optional[str] = None,
    ) -> Union[Tuple[Response, Response], Tuple[Responses, Responses]]:
        with self.connect(
            scp=False, username=username, ssh_key_file=ssh_key_file
        ) as client:
            stdouts, stderrs = [], []
            for cmd in cmds:
                _, stdout, stderr = client.exec_command(cmd)
                stderrs.append(stderr.read().decode() or None)
                stdouts.append(stdout.read().decode() or None)

            # if we're just running one command, return
            # the outputs unvarnished
            if len(cmds) == 1:
                return stdouts[0], stderrs[0]

            # otherwise return all the outputs
            return stdouts, stderrs

    def scp(
        self,
        filename: str,
        target: Optional[str] = None,
        username: Optional[str] = None,
        ssh_key_file: Optional[str] = None,
    ) -> None:
        """
        Get a file from the remote host
        """
        with self.connect(
            scp=True, username=username, ssh_key_file=ssh_key_file
        ) as client:
            return client.get(filename, target or "")
