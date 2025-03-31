# Pirates-csg-point-calculator
This program calculates an estimation for what a ship's point total should be. It utilizes ML and the error value is approximately 1 point so it is fairly accurate.
There is a premade windows .exe program in the releases tab in github here: 
https://github.com/random-retro-gaming-projects/pirates-csg-point-calculator/releases/tag/v1

Note: The following cli and installation steps require running at least these pip installs
`pip install pandas numpy scikit-learn joblib openpyxl pyinstaller`

Also in this repository is a cli version of this application, that uses the same .pkl
It can be invoked by using 
`python pirates_cli.py --masts 3 --cargo 4 --base_move "S+L" --cannons "3S,4L" --faction "Pirates" --abilities "Captain. Fear. +1 to cannon rolls."`



To create windows .exe from pirates_gui.py, in this directory:
`python -m PyInstaller -w -F --add-data "pirates_point_model.pkl;." pirates_gui.py`


To train a model from pirates.csv, run the `train_and_save_pirates.py` script in the same directory as pirates.csv. This will create a pirates_point_model.pkl in the create_ml_model directory. If you manually train, make sure to copy the .pkl into the same directory as the pirates_gui.py

The pirates.csv itself was created from the master spreadsheet called "Pirates CSG Master Spreadsheet.xlsx"
Found here on google drive: https://pirateswithben.com/pirates-csg-master-spreadsheet/
The xlsx was converted to a csv using the parse_pirates_excel_csv.py
