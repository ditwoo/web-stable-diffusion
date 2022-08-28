
web-api:
	GUNICORN_CMD_ARGS="--keep-alive 0" \
	PYTHONPATH=web_stable_diffusion \
	uvicorn main:app --host 0.0.0.0 --port 8888 --workers 1 --reload
