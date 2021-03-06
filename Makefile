.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET= helder/churnr 
PROJECT_NAME = churnr
PYTHON_INTERPRETER = python
IS_ANACONDA=$(shell python -c "import sys;t=str('anaconda' in sys.version.lower() or 'continuum' in sys.version.lower());sys.stdout.write(t)")
#VERSION=$(shell grep "__version__" churnr/__init__.py | sed 's/[^0-9.]*\([0-9.]*\).*/\1/') 
VERSION=1.0.0
NOW=`date +"%Y%m%d_%H%M%S"`

EXP=coolexp

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
reqs: test_environment
	pip install -r requirements.txt
	python setup.py develop

## Make Dataset
data: package
	gcloud compute copy-files dist/churnr-$(VERSION).tar.gz  helder-gpu-1:~/churnr --project "deep-learning-1216" --zone "us-central1-b"
	gcloud compute ssh helder-gpu-1 --project "deep-learning-1216" --zone "us-central1-b" -- -n -f 'rm -fr ~/venv > ~/job.log 2>&1; virtualenv ~/venv >> ~/job.log 2>&1; source ~/venv/bin/activate >> ~/job.log 2>&1; pip install ~/churnr/churnr-$(VERSION).tar.gz >> ~/job.log 2>&1; nohup churnr --experiment $(EXP) --stages sample parse extract process >> ~/job.log 2>&1 &'

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Create a Python package
package:
	$(PYTHON_INTERPRETER) setup.py sdist

## Download models predictions from GCS
download:
	mkdir -p models/$(EXP)
	gsutil -m cp -n -r gs://helder/churnr/$(EXP)/* models/$(EXP)

## Submit a training job to CloudML
submit: package
	gcloud ml-engine jobs submit training helderm_$(EXP)_$(NOW) --config config.yaml --job-dir gs://helderm/churnr/ --runtime-version 1.0 --module-name churnr.submitter --region us-east1 --packages dist/churnr-$(VERSION).tar.gz,libs/imbalanced-learn-0.3.0.dev0.tar.gz --project user-lifecycle -- --experiment $(EXP)

## Set up python interpreter environment
env:
ifeq (True,$(IS_ANACONDA))
		@echo ">>> Detected Anaconda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3.5
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                                #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) == Darwin && echo '--no-init --raw-control-chars')
