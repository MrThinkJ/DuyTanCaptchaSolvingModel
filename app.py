from flask import Flask, request
from flask_cors import CORS
import requests
from model import CaptchaModel
import torch
from inference import inference

app = Flask(__name__)
CORS(app)
categories = [
    "1",
    "2",
    "3",
    "4",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "N",
    "P",
    "Q",
    "R",
    "T",
    "U",
    "V",
    "X",
    "Y",
    "Z",
]
checkpoint = torch.load("trained_model/best.pt", map_location=torch.device("cpu"))
model_state = checkpoint["state"]
model = CaptchaModel(num_classes=len(categories))
model.load_state_dict(model_state)
res = requests.get("https://mydtu.duytan.edu.vn/Signin.aspx")
cookies = res.cookies.get_dict()["ASP.NET_SessionId"]
headers = {
    "cookie": "ASP.NET_SessionId={}".format(cookies),
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
}


@app.route("/process", methods=["POST"])
def process_image():
    image_src = request.data.decode("utf-8")
    process_image_source(image_src=image_src)
    output_text = inference("data/img.png", model=model, categories=categories)
    return output_text


def process_image_source(image_src):
    data = requests.get(image_src, headers=headers).content
    f = open("data/img.png", "wb")
    f.write(data)
    f.close()


if __name__ == "__main__":
    app.run()
