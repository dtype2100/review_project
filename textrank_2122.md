#### 2021, 2022년 textrank

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
df_date = df[df['날짜'].str.startswith(('2021','2022'))]
df_date
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
      <th>0</th>
      <td>0</td>
      <td>2022년 2월 7일</td>
      <td>1</td>
      <td>2/7 추가: 최신 버전 역시 여전히 느립니다. 피드백 반영도 안 하시면서 의견은 ...</td>
      <td>238</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2022년 2월 15일</td>
      <td>1</td>
      <td>앱이 너무 느림, 리뷰달때 확인도 없이 바로 달림 길게써서 포인트 받을라 했는데 못...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2022년 2월 8일</td>
      <td>1</td>
      <td>진짜 별 하나도 아깝다.. 내가 내 돈주고 물건사려고 하는데 어플이 너무 느려서 장...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2022년 2월 8일</td>
      <td>1</td>
      <td>아 진짜 뭐하자는건지 모르겠네요 화나서 리뷰 쓰러 왔더니 많은 분들이 이미 지적을 ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2022년 2월 4일</td>
      <td>1</td>
      <td>왜 별점이 2점대지..?싶을정도로 혜택도 많고 개좋은데 ㅜㅜㅜㅠ ---&gt;원래 별점 ...</td>
      <td>11명</td>
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
      <th>2155</th>
      <td>2155</td>
      <td>2021년 7월 2일</td>
      <td>5</td>
      <td>너무 느리다.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2165</th>
      <td>2165</td>
      <td>2021년 6월 11일</td>
      <td>2</td>
      <td>너무느림....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2166</th>
      <td>2166</td>
      <td>2022년 1월 9일</td>
      <td>5</td>
      <td>굿</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2169</th>
      <td>2169</td>
      <td>2021년 11월 21일</td>
      <td>5</td>
      <td>좋아요</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2170</th>
      <td>2170</td>
      <td>2021년 6월 14일</td>
      <td>5</td>
      <td>땡큐</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>950 rows × 5 columns</p>



##### 2\) 리뷰만 모아서 단어 사전 만들기

```python
df_1 = df_date['내용']
```

```python
docs = df_1.tolist()
```

```python
doc_1 = (",").join(docs)
```

```python
words = [" ".join([token[0] for token in mecab.pos(doc_1) if token[1].strip() in ["NNG", "NNP", "VV"]])]
```

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer()
word_count_vector = count_vect.fit_transform(words)
```

```python
count_vect.vocabulary_
```

```html
{'추가': 948,
 '최신': 942,
 '버전': 412,
 '립니': 296,
 '피드백': 1068,
 '반영': 386,
 '의견': 736,
 ...}
```



##### 3\) tfidf & textrank

```python
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(word_count_vector)
```

```python
TfidfTransformer()
```

```python
def sort_keywords(keywords):
    return sorted(zip(keywords.col,keywords.data),key=lambda x :(x[1],x[0]), reverse=True)

def extract_keywords(feature_name, sorted_keywords,n=10):
    return [(feature_name[idx],score) for idx,score in sorted_keywords[:n]]
```

```python
feature_name = count_vect.get_feature_names_out()
tf_idf_vector = tfidf_transformer.transform(count_vect.transform(words))
sorted_keywords = sort_keywords(tf_idf_vector.tocoo())
keywords = extract_keywords(feature_name,sorted_keywords)
keywords
```

```html
[('업데이트', 0.2835631651497142),
 ('리뷰', 0.2752230720570755),
 ('배송', 0.272443041026196),
 ('사용', 0.21128235834684586),
 ('구매', 0.19738220319244812),
 ('쿠폰', 0.19460217216156855),
 ('광고', 0.18904211009980945),
 ('화면', 0.18626207906892991),
 ('불편', 0.1751419549454117),
 ('상품', 0.15568173772925484)]
```

