import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import paramiko
from rich.progress import Progress
from scp import SCPClient

from hermes.cloudbreak import logger
from hermes.cloudbreak.base.resource import Resource

Response = Optional[str, None]
Responses = List[Response]


# TODO: have this inherit from ManagerResource
class VMManager:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self._resources = {}

    @property
    def client(self):
        return self.parent.client

    def create_one_vm(self, name):
        raise NotImplementedError

    @contextmanager
    def manage(self, N):
        self.create(N)
        self.wait_for_ready()

        try:
            yield self
        finally:
            self.delete()
            self.wait_for_delete()

    def delete(self):
        for name, vm in self._resources.items():
            vm.delete()

    def wait_for_delete(self):
        with Progress() as progbar:
            resources = self._resources.copy()
            task_id = progbar.add_task(
                "Waiting for VMs to be ready", total=len(self._resources)
            )
            while self._resources:
                for name in resources.keys():
                    try:
                        vm = resources[name]
                    except KeyError:
                        continue
                    if vm.is_deleted():
                        self._resources.pop(name)
                        progbar.update(task_id, advance=1)

    def create(self, N: int, username: str, ssh_key_file: str):
        for i in range(N):
            name = self.name + f"-{i}"
            try:
                vm = self._resources[name]
            except KeyError:
                vm = self.create_one_vm(name)

            vm.set_user(username, ssh_key_file)
            self._resources[name] = vm

    def wait_for_ready(self):
        with Progress() as progbar:
            resources = self._resources.copy()
            task_id = progbar.add_task(
                "Waiting for VMs to be ready", total=len(self._resources)
            )
            while resources:
                for name in self._resources.keys():
                    try:
                        vm = resources[name]
                    except KeyError:
                        continue
                    if vm.is_ready():
                        resources.pop(name)
                        progbar.update(task_id, advance=1)


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

    @classmethod
    def create(self, name, *args, **kwargs):
        raise NotImplementedError

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
                    hostname=self.instance.ip,
                    key_filename=ssh_key_file,
                )
                break
            except Exception as e:
                logger.warning(
                    "Failed to connect to VM {} at IP {} on attempt {} "
                    "after receiving error {}. Trying again".format(
                        self.instance.name, self.instance.ip, i + 1, str(e)
                    )
                )
                time.sleep(1)
        else:
            # if after 5 attempts we were unable to
            # connect to the instance, raise an error
            raise RuntimeError(
                "Failed to connect to VM {} at IP {} after 5 attempts".format(
                    self.instance.name, self.instance.ip
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
