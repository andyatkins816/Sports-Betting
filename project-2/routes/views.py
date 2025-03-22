from flask import Blueprint, render_template
import logging

logger = logging.getLogger(__name__)
bp = Blueprint('views', __name__)

@bp.route('/')
def index():
    """View for home page"""
    logger.info("Accessing home page")
    return render_template('index.html')

@bp.route('/predictions')
def predictions():
    """View for predictions page"""
    logger.info("Accessing predictions page")
    try:
        return render_template('predictions.html')
    except Exception as e:
        logger.error(f"Error rendering predictions page: {e}", exc_info=True)
        raise

@bp.route('/dashboard')
def dashboard():
    """View for prediction accuracy dashboard"""
    logger.info("Accessing dashboard page")
    return render_template('dashboard.html')