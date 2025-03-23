import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

def create_app():
    # Create Flask app
    app = Flask(__name__)

    # Configure app
    app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }

    # Initialize extensions
    db.init_app(app)

    with app.app_context():
        # Import and register blueprints
        from routes.views import bp as views_bp
        app.register_blueprint(views_bp)

        from routes.api import bp as api_bp
        app.register_blueprint(api_bp)

        # Create database tables
        db.create_all()

        # Log registered routes for debugging
        logger.debug("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.debug(f"{rule.endpoint}: {rule.rule}")

        return app

# Create the application instance
app = create_app()