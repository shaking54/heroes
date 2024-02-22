## SOLUTION BRIEF

I see that I don't have the training dataset to train supervised classifier. So that, I try to use template matching algorithms to solve this problem.

- Step 1: I collect all image of heroes in Leauge of Legends (icons image). I call it template image.
- Step 2: I extract all these template image into features using LightGlue and save it as local variable 
- Step 3: When an input image get in, I will using LightGlue to extract features and using a matcher to match these 2 features and find the match points in two images. I save the number match points as a score to choose the label.
- Step 4: I choose the template image has the max match points as the label of input image.

```
- Advantages: I don't have to collect any data and train the model.
- Disadvantages: + Performance is still unacceptable (for me above 80%). 
                 + The inference is very slow ~ 1 minutes for a pairs of image. In lightGlue, the author have paramters to trade off between accuracy and latency. And I choose optimize accuracy first.  
```
### STEP 0: GET TEMPLATE OF HEROES

install npm with this command
```
sudo su -c 'curl -sL https://deb.nodesource.com/setup_18.x | bash -'
sudo apt-get install nodejs -y
sudo apt update
sudo apt upgrade
sudo npm install -g npm@10.4.0
node --version
npm --version
```

install this repo to download image template https://github.com/Hi-Ray/cd-dd
```
npm i cdragon-dd -g
```

download template hero images
```
cd-dd https://raw.communitydragon.org/14.3/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/
```

### STEP 1: CREATE ENVIRONMENTS

using conda
```
conda create -n heroes python=3.9
conda activate heroes
pip install -r requirements
```

### STEP 2: INSTALL LIGHTGLUE

```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

### STEP 3: RUN INFERENCE.
list of argument
- options : enum ["inference", "evaluate"]
- image : "image/path"
- depth_confidence: float
- width_confidence: float

```
python main.py --options inference --image "image/path" --depth_confidence -1 --width_confidence -1
```

### STEP 4: EVALUTATE.

```
python main.py --options evaluate --test_data_path test_data/test_images/ --ground_truth test_data/test.txt
```
