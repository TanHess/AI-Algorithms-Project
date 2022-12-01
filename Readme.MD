# Baseball Predictor App


## Setup Instructions

The following instructions lay out how to settup the Baseball Predictor App. If there are issues while following these instructions, please contact one of the contributors of this project for assistance. 

#### IMPORTANT NOTES
- If the python/pip commands listed within these instructions are not working, replace “python” and “pip” with “python3” and “pip3” respectively.
- To download and setup this project, you must first have python along with pip (the package manager for python) installed on your device. 
  - If you do not have python, please follow this link to download it https://www.python.org/downloads/. 
  - After installing python, ensure pip installed along with your package and is working properly by running the command “pip --version” in the command line (most IDE’s have a built-in command line, if not, use your system’s command line). 

After ensuring python and pip are installed, follow these step-by-step instructions to download this project:
1.	Create a directory to store the project files in wherever suites you.
2.	Go to https://github.com/TanHess/AI-Algorithms-Project
3.	Click the green “Code” button on the page 
4.	Click the “Download Zip” button
5.	Move the downloaded zip file to your directory you created earlier
6.	Unzip the project
7.	Use either the command line or an IDE with a command line to navigate to the project directory.
8.	Once in the project directory within the command line, run the command “pip install -r  requirements.txt” 
9.	After installing the dependencies, run the command “python main.py” to run the web application.
    - Travel to localhost:5000 (127.0.0.1:5000) in a browser to view the web application.
    - To stop the application, use the terminal you started it from and simultaneously press “control-c”

## Using the Baseball Pitch Predictor:

When you arrive at the web page hosting the GUI for the Baseball Predictor, you will be met with a screen like this:
<img width="1728" alt="image" src="https://user-images.githubusercontent.com/89549856/205119092-00494672-31cc-49ef-afb8-c631552c327c.png">

To use the pitch predictor, enter the appropriate information in each field. If you are uncertain what a field means, hover your cursor over the title of the field for a tooltip further explaining what to enter:<img width="862" alt="image" src="https://user-images.githubusercontent.com/89549856/205120783-fb7cfb95-1e28-46b3-8fb6-90a947241c51.png">

Finally, after entering the current standings to the pitch predictor, click "Predict the Pitch" to recieve your result. The given result will indicate the distrubution of pitches the Pitcher has thrown during their career (left graph), as well as the odds of each of those pitches for the current scenario (right graph/prediction). Additionally, the Pitch Predictor will display the most likely pitch, the pitch that most significantly increased in likelyhood of being thrown, and the pitch that most significantly decreased in likelyhood of being thrown (based on the distribution of actual thrown pitches).

### Example of Pitch Predictor Running:

<img width="1728" alt="image" src="https://user-images.githubusercontent.com/89549856/205122057-55b466d9-d2f9-412c-8fd7-e6633e7612dd.png">
<img width="1726" alt="image" src="https://user-images.githubusercontent.com/89549856/205122107-1dfb59da-5904-41e7-bf87-58bc46aceac5.png">

