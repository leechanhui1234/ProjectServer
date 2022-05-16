from flask import Flask, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

df = pd.read_csv('exercise.csv', encoding="cp949")
data=df[['운동 이름', '유/무산소', '운동 부위', '도구', '단체유무', '장소', '나이', '비만도', '성별', '상장', '운동순서', '순위']]

counter_vector = CountVectorizer(ngram_range=(1, 3))

def recommend(df, text, ex_body, top=30):
    c_vector_genres = counter_vector.fit_transform(df[text])
    similarity_genre = cosine_similarity(c_vector_genres, c_vector_genres).argsort()[:, ::-1]
    target_index = df[df[text].str.contains(ex_body)].index.values
    sim_index = similarity_genre[target_index,:top].reshape(-1)
    result = df.iloc[sim_index][:int(df.shape[0] * 0.9)]
    return result

def tt(geninfo, ageinfo, bodyinfo, thinginfo, heightinfo, weightinfo):
    gen = geninfo
    age = ageinfo
    d = bodyinfo
    things = thinginfo + ',없음'
    height = float(heightinfo)
    weight = float(weightinfo)
    bmi = weight / ((height / 100) ** 2)
    if (bmi < 18.5):
        bm = "저체중" + ",정상"
    elif (bmi >= 18.5 and bmi <= 22.9):
        bm = "정상" + ",저체중,비만전단계비만"
    elif (bmi >= 23 and bmi <= 24.9):
        bm = "비만전단계비만" + ",정상,1단계비만"
    elif (bmi >= 25 and bmi <= 29.9):
        bm = "1단계비만" + ",비만전단계비만,2단계비만"
    elif (bmi >= 30 and bmi <= 34.9):
        bm = "2단계비만" + ",1단계비만,3단계비만"
    else:
        bm = "3단계비만" + ",2단계비만"

    data = df[['운동 이름', '유/무산소', '운동 부위', '도구', '단체유무', '장소', '나이', '비만도', '성별', '상장', '운동순서', '순위']]
    gender_data = df[df['성별'] == gen]
    age_data = gender_data[gender_data['나이'] == age]

    pre_data = age_data[age_data['운동순서'] == '준비운동']
    pre_data.reset_index(inplace=True)
    in_data = age_data[age_data['운동순서'] == '본운동']
    in_data.reset_index(inplace=True)
    post_data = age_data[age_data['운동순서'] == '마무리운동']
    post_data.reset_index(inplace=True)

    bm = bm.split(",")
    index = 0
    for i in bm:
        if index == 0:
            pre_bmi = pre_data[pre_data['비만도'] == i]
            in_bmi = in_data[in_data['비만도'] == i]
            post_bmi = post_data[post_data['비만도'] == i]
            index = index + 1
        else:
            pre_bmi = pd.concat([pre_bmi, pre_data[pre_data['비만도'] == i]])
            in_bmi = pd.concat([in_bmi, in_data[in_data['비만도'] == i]])
            post_bmi = pd.concat([post_bmi, post_data[post_data['비만도'] == i]])

    pre_bmi.reset_index(inplace=True, drop=True)
    in_bmi.reset_index(inplace=True, drop=True)
    post_bmi.reset_index(inplace=True, drop=True)

    things = things.split(",")
    index = 0
    for i in things:
        if index == 0:
            pre_things = pre_bmi[pre_bmi['도구'] == i]
            in_things = in_bmi[in_bmi['도구'] == i]
            post_things = post_bmi[post_bmi['도구'] == i]
            index = index + 1
        else:
            pre_things = pd.concat([pre_things, pre_bmi[pre_bmi['도구'] == i]])
            in_things = pd.concat([in_things, in_bmi[in_bmi['도구'] == i]])
            post_things = pd.concat([post_things, post_bmi[post_bmi['도구'] == i]])

    pre_things.reset_index(inplace=True, drop=True)
    in_things.reset_index(inplace=True, drop=True)
    post_things.reset_index(inplace=True, drop=True)

    d = d.split(",")

    index = 0

    for i in d:
        if index == 0:
            pre_body = recommend(pre_things, "운동 부위", i)
            in_body = recommend(in_things, "운동 부위", i)
            post_body = recommend(post_things, "운동 부위", i)
            pre_answer1 = pre_things[pre_things['운동 부위'].str.contains(i)]
            in_answer1 = in_things[in_things['운동 부위'].str.contains(i)]
            post_answer1 = post_things[post_things['운동 부위'].str.contains(i)]
            pre_body.reset_index(inplace=True, drop=True)
            in_body.reset_index(inplace=True, drop=True)
            post_body.reset_index(inplace=True, drop=True)
            index = index + 1
        else:
            pre_body = pd.concat([pre_body, recommend(pre_things, "운동 부위", i)])
            in_body = pd.concat([in_body, recommend(in_things, "운동 부위", i)])
            post_body = pd.concat([post_body, recommend(post_things, "운동 부위", i)])
            pre_answer1 = pd.concat([pre_answer1, pre_things[pre_things['운동 부위'].str.contains(i)]])
            in_answer1 = pd.concat([in_answer1, in_things[in_things['운동 부위'].str.contains(i)]])
            post_answer1 = pd.concat([post_answer1, post_things[post_things['운동 부위'].str.contains(i)]])
            pre_body.reset_index(inplace=True, drop=True)
            in_body.reset_index(inplace=True, drop=True)
            post_body.reset_index(inplace=True, drop=True)

    pre_answer1 = pre_answer1.drop_duplicates()  # 필터링 추천
    in_answer1 = in_answer1.drop_duplicates()  # 필터링 추천
    post_answer1 = post_answer1.drop_duplicates()  # 필터링 추천

    pre_body = pre_body.drop_duplicates(['운동 이름'])  # 유사도 추천
    in_body = in_body.drop_duplicates(['운동 이름'])  # 유사도 추천
    post_body = post_body.drop_duplicates(['운동 이름'])  # 유사도 추천

    str1 = pre_answer1["운동 이름"].sample(n=1).values
    str2 = pre_body["운동 이름"].head(3).values

    if (str1[0] == str2[0]):
        str2[0] = str2[1]
        str2[1] = str2[2]
    elif (str1[0] == str2[1]):
        str2[1] = str2[2]

    str3 = in_answer1["운동 이름"].sample(n=1).values
    str4 = in_body["운동 이름"].head(3).values

    if (str3[0] == str4[0]):
        str4[0] = str4[1]
        str4[1] = str4[2]
    elif (str3[0] == str4[1]):
        str4[1] = str4[2]

    str5 = post_answer1["운동 이름"].sample(n=1).values
    str6 = post_body["운동 이름"].head(3).values

    if (str5[0] == str6[0]):
        str6[0] = str6[1]
        str6[1] = str6[2]
    elif (str5[0] == str6[1]):
        str6[1] = str6[2]

    str = str1[0] + "," + str2[0] + "," + str2[1] + "|" + str3[0] + "," + str4[0] + "," + str4[1] + "|" + str5[0] + "," + str6[0] + "," + str6[1]
    return str

@app.route("/", methods=['GET', 'POST'])
def handle_request():
    data = request.data.decode('utf-8')
    test = data.split("|")
    gen = test[0]
    age = test[1]
    body = test[2]
    thing = test[3]
    leng = test[4]
    weight = test[5]
    d = tt(gen, age, body, thing, leng, weight)
    print(d)
    return d



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)