
web-api:
	PYTHONPATH=web_stable_diffusion uvicorn main:app --host 0.0.0.0 --port 8888 --workers 1 --reload
