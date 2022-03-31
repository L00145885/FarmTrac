from flask import send_from_directory
from website import create_app

application = create_app()

@application.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)
    
if __name__ == '__main__':
    application.run(port='8080')