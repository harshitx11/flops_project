"""Development entry point for the Review Sentiment MLOps project."""

import uvicorn

from sentiment_mlops.api.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
