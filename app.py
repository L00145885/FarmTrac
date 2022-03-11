from flask import send_from_directory
from website import create_app

app = create_app()

@app.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)
    
if __name__ == '__main__':
    app.run(debug=True)