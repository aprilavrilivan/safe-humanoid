PYTHON ?= python3

.PHONY: test test-sim doctor smoke smoke-safety smoke-scenarios train

test:
	$(PYTHON) -m unittest discover -s tests/unit -v

test-sim:
	$(PYTHON) -m unittest discover -s tests/sim -v

doctor:
	$(PYTHON) scripts/doctor.py

smoke:
	$(PYTHON) scripts/smoke_env.py --headless --num-envs 4 --steps 32

smoke-safety:
	$(PYTHON) scripts/smoke_safety.py --headless --num-envs 2 --steps 64

smoke-scenarios:
	$(PYTHON) -m unittest discover -s tests/sim -p test_scenarios.py -v

train:
	$(PYTHON) scripts/train.py --scenario nominal --num-envs 512
