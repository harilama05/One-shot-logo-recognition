import os
from flask import Blueprint, render_template, request, Response, send_from_directory, make_response
from config import Config

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    return render_template('index.html')

@views_bp.route('/favicon.ico')
def favicon():
    return '', 204

@views_bp.after_request
def add_no_cache_headers(resp):
    if request.path.startswith('/detected'):
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
    return resp

@views_bp.route('/detected/<filename>')
def serve_detected_file_alt(filename):
    full_path = os.path.join(Config.DETECTED_DIR, filename)
    if not os.path.exists(full_path):
        transparent_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x18\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
        return Response(transparent_png, mimetype='image/png', headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0'
        })
    resp = make_response(send_from_directory(Config.DETECTED_DIR, filename))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp
