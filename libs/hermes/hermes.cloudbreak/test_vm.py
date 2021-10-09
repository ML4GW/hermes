import os
from pathlib import Path

from hermes.cloudbreak.clouds import google as cb

home = Path.home()
with open(home / ".bash_aliases", "r") as f:
    for row in f.read().splitlines():
        alias, value = row.split("=")
        os.environ[alias] = value


description = cb.make_simple_debian_instance_description(
    name="test-vm",
    zone="us-west1-b",
    vcpus=4,
    startup_script="echo 'hello world!'",
)

manager = cb.VMManager(description)
with manager.manage(
    40, username="alec.gunny", ssh_key_file=home / ".ssh" / "id_rsa.gcloud"
):
    cmd = """
        sudo apt-get install -y git && \
        git clone https://github.com/alecgunny/gw-iaas && \
        cd gw-iaas && \
        cat README.md && \
        echo {i}
    """
    stdouts, stderrs = manager.run(cmd, i=[i for i in range(40)])
    print(stdouts["test-vm-0"])
