import os
import datetime
from ppadb.client import Client as AdbClient

class Photography():
    def __init__(self):
        self.client = AdbClient(host="127.0.0.1", port=5037)
        self.device = self.client.device("emulator-5554")

    def takePicture(self):
        self.result = self.device.screencap()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"screen_{timestamp}.png"

        with open(filename, "wb") as fp:
            fp.write(self.result)

        return self.result, filename
    
    def deletePicture(self, filename):
        wd = os.getcwd()
        file_path = os.path.join(wd, filename)

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File deleted successfully: {file_path}")
            else:
                print(f"Error: The file {file_path} does not exist.")
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}")