import requests

res = requests.get("https://mydtu.duytan.edu.vn/Signin.aspx")
print(res.cookies.get_dict()["ASP.NET_SessionId"])
