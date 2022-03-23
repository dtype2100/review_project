#### ~2020년 textrank

##### 1\) 데이터 불러오기

```python
import pandas as pd
text = pd.read_csv("olive_app_review_crawling_final.csv")
```

```python
from konlpy.tag import Mecab
mecab = Mecab()
```

```python
df = pd.DataFrame(text)
```

```python
df_date1 = df[df['날짜'].str.startswith(('201','2020'))]
df_date1
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>날짜</th>
      <th>별점</th>
      <th>내용</th>
      <th>추천</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>159</th>
      <td>159</td>
      <td>2020년 3월 3일</td>
      <td>1</td>
      <td>앱이 로그인이안되네요~ 소프트웨어 ㆍ인터넷 최신 업데이트 했는데도 폰 인터넷홈페이지...</td>
      <td>5명이</td>
    </tr>
    <tr>
      <th>160</th>
      <td>160</td>
      <td>2019년 7월 1일</td>
      <td>1</td>
      <td>진짜 돌아버리겠네요, 결제하려고 하면 왜 메인화면으로 돌아가나요...? 10번은 넘...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>161</th>
      <td>161</td>
      <td>2018년 9월 21일</td>
      <td>5</td>
      <td>직접 올리브영으로 가지않아도 집에서도 손쉽게 쇼핑할 수 있어서 정말 좋아요!! 덕분...</td>
      <td>117</td>
    </tr>
    <tr>
      <th>164</th>
      <td>164</td>
      <td>2020년 2월 12일</td>
      <td>1</td>
      <td>갑자기 잘되던 앱이 로그인도 안되고.삭제하고 다시 설치했는데. 로그인이 된 척하면서...</td>
      <td>4명이</td>
    </tr>
    <tr>
      <th>166</th>
      <td>166</td>
      <td>2018년 9월 16일</td>
      <td>1</td>
      <td>어제부터 계속 결제할 때 결제창은 안 뜨고 사용할 앱을 선택하라고 뜨네요 원래는 안...</td>
      <td>72명</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2163</th>
      <td>2163</td>
      <td>2019년 11월 25일</td>
      <td>5</td>
      <td>좋아요</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2164</th>
      <td>2164</td>
      <td>2018년 10월 15일</td>
      <td>4</td>
      <td>좋아요</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2167</th>
      <td>2167</td>
      <td>2016년 8월 3일</td>
      <td>5</td>
      <td>최고</td>
      <td>1명이</td>
    </tr>
    <tr>
      <th>2168</th>
      <td>2168</td>
      <td>2020년 10월 9일</td>
      <td>5</td>
      <td>좋아요</td>
      <td>1명이</td>
    </tr>
    <tr>
      <th>2171</th>
      <td>2171</td>
      <td>2014년 10월 12일</td>
      <td>5</td>
      <td>좋아요</td>
      <td>1명이</td>
    </tr>
  </tbody>
</table>
<p>1222 rows × 5 columns</p>



##### 2\) 리뷰만 모아서 단어 사전 만들기

```python
df_11 = df_date1['내용']
```

```python
docs1 = df_11.tolist()
```

```python
doc_11 = (",").join(docs1)
```

```python
words = [" ".join([token[0] for token in mecab.pos(doc_11) if token[1].strip() in ["NNG", "NNP", "VV"]])]
```

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer()
word_count_vector = count_vect.fit_transform(words)
```

```pyth
count_vect.vocabulary_
```

```html
{'로그인': 273,
 '소프트웨어': 544,
 '최신': 954,
 '업데이트': 645,
 '인터넷': 781,
 '홈페이지': 1115,
 '비번': 466,
 ...}
```



##### 3\) tfidf & textrank

```python
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(word_count_vector)
```

```html
TfidfTransformer()
```

```python
def sort_keywords(keywords):
    return sorted(zip(keywords.col,keywords.data),key=lambda x :(x[1],x[0]), reverse=True)

def extract_keywords(feature_name, sorted_keywords,n=10):
    return [(feature_name[idx],score) for idx,score in sorted_keywords[:n]]
```

```python
eature_name = count_vect.get_feature_names_out()
tf_idf_vector = tfidf_transformer.transform(count_vect.transform(words))
sorted_keywords = sort_keywords(tf_idf_vector.tocoo())
keywords = extract_keywords(feature_name,sorted_keywords)
keywords
```

```html
[('로그인', 0.6231246732172319),
 ('결제', 0.25098077115694056),
 ('쿠폰', 0.1990537150555046),
 ('올리브', 0.19559191131540887),
 ('가입', 0.19039920570526528),
 ('오류', 0.17135928513473875),
 ('사용', 0.16616657952459515),
 ('회원', 0.1540502664342601),
 ('화면', 0.1540502664342601),
 ('구매', 0.15231936456421222)]
```

