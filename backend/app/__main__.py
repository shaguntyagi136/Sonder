"""python -m app  -> runs the API using API_HOST / API_PORT from the environment."""

import uvicorn



from app.core.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("app.main:create_app", factory=True, host=s.api_host, port=s.api_port, reload=s.app_debug)
