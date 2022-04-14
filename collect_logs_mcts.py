import subprocess
import time
import glob
import logging


cases = glob.glob("test-cases/test/*")  # glob.glob("test-cases/train/*")
cases = sorted(cases, reverse=False)

switches = [0]  # [0,1,2,3,4]

logging.basicConfig(
    filename="logs_grasp/collect.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logger.info(f"Starting collect mcts data...")
logger.info(f"Switch: {switches}")
logger.info(f"Cases {cases}")

commands = []
for switch in switches:
    for case in cases:
        file_path = case
        c = ["python", "mcts_main.py", "--test_case", file_path, "--max_test_trials", "5", "--test"]
        if switch != 0:
            c.extend(["--switch", str(switch)])
        commands.append(c)
        print(c)

print("=================================================")


max_procs = 6
procs = []
names = []
while len(commands) > 0:
    if len(procs) < max_procs:
        command = commands.pop(0)
        print(f"Staring: {command}")
        procs.append(subprocess.Popen(command))
        names.append(command)
        logger.info(f"Staring {command}")

    for p in procs:
        poll = p.poll()
        if poll is not None:
            idx = procs.index(p)
            procs.pop(idx)
            info = names.pop(idx)
            print(f"End: {command}")
            logger.info(f"End {info}")

    time.sleep(5)

