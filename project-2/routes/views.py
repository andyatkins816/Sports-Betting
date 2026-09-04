from flask import Blueprint, abort, render_template
bp = Blueprint('views', __name__)

@bp.route('/')
def index():
    """Public status page; no model or betting claims are displayed."""
    return render_template('index.html')

@bp.route('/predictions')
def predictions():
    """Retire the demo UI until it is backed by licensed, audited data."""
    abort(410, description="The demo prediction UI is retired pending verified data integration.")

@bp.route('/dashboard')
def dashboard():
    """Retire the demo dashboard that could display simulated performance."""
    abort(410, description="The demo dashboard is retired pending verified historical evaluation.")
