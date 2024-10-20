# Fire Detection Project Description

Our project topic is fire detection. We have sourced a collection of fire vs non-fire images from the [fire dataset, Kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset  ). We plan on using supervised learning for this project for the use of classifying images. From the dataset we are going to use 80% of our dataset as training data and 20% as testing. The practical use for this project is to detect fire from imaging using satellite’s or trail cams/ cameras on towers.

We got inspiration from this article on how our detection model might be useful: [www.washingtonpost.com](https://www.washingtonpost.com/climate-solutions/2023/06/15/wildfire-early-detection-sensors-technology/). In this post Micheal Pavolonis, the Wildland Fire Program manager at NOAA’s National Environmental Satellite, Data and Information Service, says “Every technology and technique has strengths and limitations,” and “There isn’t a single technique that can solve this problem.” In these quotes he is referring to how smoke detection sensors, satellite imaging or trail cameras can't solve the wildfire issue alone. All of these solutions instead need to be implemented together to create a strong fail-safe system. 

With this being said our fire-detection model can’t holistically solve the growing problem of wildfires in areas that are prone to it, but can be a great tool in connection with others to solve the issue at hand. We are going to use the CNN named [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) as it is optimized to allow users to choose between latency and accuracy with two basic hyperparameters. This CNN also was designed for mobile and embedded vision applications so this would be a great CNN to use for our desired use. 


# How to Run the Project
1. Clone this repository 
    - `git clone git@github.com:David-Tincher/fire-detection.git`
2. Navigate to the project directory 
    - `cd fire-detection`
3. Set up the Python virtual environment
    - `python -m venv .venv` <br />
    - For Linux: `source .venv/bin/activate` <br />
      For Windows: `.venv\scripts\activate\`
4. Install dependencies
    - `pip install -r requirements.txt`
