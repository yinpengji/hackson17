# Comify video for hackathon 2017


## Usage

### Extract scene
  run with python 3.5
  python scene_detection.py -s /path/to/input/file -d /path/to/output/folder -n identifierName 
  
  for example: python scene_detection.py -s "c:\wonder_1.mp4" -d C:\videoextract -n thumnail
  
  The extracted scene will be stored under C:\videoextract\images\full 
  
  The folder images and full is created by the script.

##### How it works
  They python will extract the scene by comparing each frame, once the change is big enough, we'll think it's a new scene.

### Get caption for each frame
  First create a C# project, it will needs NewtonJson, so after you created the project, you need to add reference to Newton
  
  It will ask for the input folder for all the images you want to caption. After execution, it will write an html file to C:\videoextract\stroy.html


### Dependency
  OpenCV 3.2
  
  Newtonsoft.Json
