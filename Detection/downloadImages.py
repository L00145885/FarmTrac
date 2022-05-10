#script to download non-cow image
import requests

path = "C:/Users/david/OneDrive - Letterkenny Institute of Technology/Year4/Semester Two/Project Development/Detection/Dataset/no_cow"

for i in range(769):
    url = "https://picsum.photos/500/500/?random"
    response = requests.get(url)
    if response.status_code == 200:
        file_name = 'not_cow_{}.jpg'.format(i)
        file_path = path + "/" + file_name
        with open(file_path, 'wb') as f:
            print("saving: " + file_name)
            f.write(response.content)
            