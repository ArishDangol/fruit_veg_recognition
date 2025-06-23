# fruit_veg_recognition
This project is Fruits and Recognition using Custom CNN as Deep Learning Project made using Python, PyTorch, Pandas, Numpy, and others.  and finally the output is displayed in the web app made using Streamlit.

Steps to Run the Project:
1) Download the project as zip from github and unzip the folder.
2) Then open that unzipped folder in the code editor (VS Code) and open the terminal.
3) Create a virtual enviroment so that python version and packages won't conflict in the future, here is the command to run first , "pip install virtualenv" this installs the vritual env and now to create virtual environment here is the command "python -m venv <virtual-environment-name>" , eg: python -m venv myvenv.
4) after creating the virtual env we need to activate it so that we can use it, use this command "source env/bin/activate" , here is the eg : env/Scripts/activate.bat //In CMD
 env/Scripts/Activate.ps1 //In Powershel, after activating then go to next step.
5) first we need to download all the requiremnt packages and modules, run the command "pip install -r requirements.txt, this command installs everyting that is required to run the project, it might take some time to download maybe few minutes so be patient on this.
6) After getting all the requirments, we then nned to train the model, open the scripts folder and find the train.py under scrips/train.py, in the terminal use this command "python -m scripts.train" , this will train the model and this takes time around 2-5 hours depending upon if CPU is used for GPU , processor dependent.
7) after traning , evaluation of model is necessary so run this command after traning "python -m scripts.evaluate",  a new window pops up showing a confusion matrix on the graph and many more charts, and in the termnial f1 score, accuracy , precision, support are seen, that are used for evaluation of model.
8) after evaluating, we can get the output in 2 ways, first way is CLI that is running the predict.py file and using cmd, second opetion is to run the streamlit GUI app file under the folder webapp/app.py
9) TO run CLI use the cmd "python -m scripts.predict <name-of-file><extenstion-type>, example: "python -m scripts.predict apple.jpg"
10) TO run the GUI based streamlit use the cmd "streamlit run webapp/app.py", then it will open in local host.
