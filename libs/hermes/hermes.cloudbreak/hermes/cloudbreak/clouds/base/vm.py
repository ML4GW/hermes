import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    def resources(self) -> Dict[str, "VMInstance"]:
        return {vm.name: vm for vm in self._resources}

    @property
    def N(self) -> int:
        return len(self._resources)

    @property
    def client(self):
        return self.parent.client

    def create_one_vm(self, name: str) -> "VMInstance":
        raise NotImplementedError

    @contextmanager
    def manage(self, N: int, username: str, ssh_key_file: str) -> "VMManager":
        self.create(N, username, ssh_key_file)
        self.wait_for_ready()

        try:
            yield self
        finally:
            self.delete()
            self.wait_for_delete()

    def delete(self) -> None:
        for vm in self._resources:
            vm.delete()

    def wait_for_delete(self) -> None:
        with Progress() as progbar:
            task_id = progbar.add_task(
                "Waiting for VMs to be deleted", total=self.N
            )
            while self._resources:
                completed = [vm for vm in self._resources if vm.is_deleted()]
                for vm in completed:
                    self._resources.remove(vm)

                progbar.update(task_id, advance=len(completed))

    def create(self, N: int, username: str, ssh_key_file: str) -> None:
        for i in range(N):
            name = f"{self.name}-{i}"
            try:
                vm = self.resources[name]
            except KeyError:
                vm = self.create_one_vm(name)
                self._resources.append(vm)

            vm.set_user(username, ssh_key_file)

    def wait_for_ready(self) -> None:
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

    def run(self, cmd: str, **kwargs: Sequence) -> Tuple[Responses, Responses]:
        """Run a command on all VMs in parallel

        Runs the specified command on all VMs being managed
        in parallel using threading. Commands can include
        formatting wildcards which will be formatted by
        any keyword arguments passed.

        Args:
            cmd:
                The command to execute on the VMs, possibly including
                wildcard values to be formatted for each VM
            **kwargs:
                Named arguments matching the wildcard values in `cmd`
                which consist of sequences of length matching the
                number of VMs. For the `i`th VM, the command executed
                on it will be formatted using the `i`th value from
                each of the keyword arguments.
        Returns:
            A tuple of dictionaries mapping VM names to
            the stdout and stderr streams from each VM
        """

        # make sure that any keyword arguments we specified
        # for formatting have the proper number of args
        for arg_name, args in kwargs.items():
            if len(args) != self.N:
                raise ValueError(
                    "Found too many values for argument {}. "
                    "Expected {}, found {}".format(arg_name, self.N, len(args))
                )

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
            for i, vm in enumerate(self._resources):
                # format the command with any arguments first
                formatted = cmd.format(
                    **{arg_name: arg[i] for arg_name, arg in kwargs.items()}
                )

                # submit the formatted command
                future = ex.submit(vm.run, formatted)
                futures.append(future)

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
    def __post_init__(self) -> None:
        self._ip = None
        self._start_time = time.time()
        self._is_ready = False

        self.username = None
        self.ssh_key_file = None

    @property
    def ip(self) -> Optional[str]:
        if self._ip is not None:
            return self._ip

    @property
    def message(self) -> str:
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
        *cmds: str,
        username: Optional[str] = None,
        ssh_key_file: Optional[str] = None,
    ) -> Union[Tuple[Response, Response], Tuple[Responses, Responses]]:
        """Run a set of commands on a remote VM instance

        Run the provided commands sequentially on this
        remote VM instance via SSH, optionally using a
        specified set of user credentials. No attempt to
        parse errors is made, so all commands will be
        executing even if previous commands fail.

        Args:
            *cmds:
                The commands to execute on the remote VM
            username:
                The user to connect to the remote VM as. If left
                as `None`, `VMInstance.username` will be used
                as the default, so ensure that this has been set
                via `VMInstance.set_user`
            ssh_key_file:
                A private ssh key used to connect to the VM. If
                left as `None`, `VMInstance.ssh_key_file` will be
                used as the default, so ensure that this has been
                set via `VMInstance.set_user`
        Returns:
            If only one command is specified, the decoded string
            values of stdout and stderr from the machine will be
            returned as a tuple, with `None` used to replace
            blank strings from either. If multiple commands are
            specified, these tuples will be returned as a list
            for each command.
        """

        with self.connect(
            scp=False, username=username, ssh_key_file=ssh_key_file
        ) as client:
            stdouts, stderrs = [], []
            for cmd in cmds:
                logger.debug(
                    "VM {}:{} executing command {}".format(
                        self.name, self.ip, cmd
                    )
                )

                _, stdout, stderr = client.exec_command(cmd)
                stdout = stdout.read().decode()
                stderr = stderr.read().decode()

                logger.debug(
                    "VM {}:{} stdout {}".format(self.name, self.ip, stdout)
                )
                logger.debug(
                    "VM {}:{} stderr {}".format(self.name, self.ip, stderr)
                )

                stderrs.append(stdout or None)
                stdouts.append(stderr or None)

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
