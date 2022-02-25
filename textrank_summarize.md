1. 각 리뷰 textrank



```python
import pandas as pd
text = pd.read_csv("olive_app_review_crawling_final.csv")
```



```python
df = pd.DataFrame(text)
df['내용']
```

> ```
> 0       2/7 추가: 최신 버전 역시 여전히 느립니다. 피드백 반영도 안 하시면서 의견은 ...
> 1       앱이 너무 느림, 리뷰달때 확인도 없이 바로 달림 길게써서 포인트 받을라 했는데 못...
>                                        ...                        
> 
> 2170                                       땡큐
> 2171                                       좋아요
> Name: 내용, Length: 2172, dtype: object
> ```



```python
docs = df['내용'].tolist()
docs
```

> ```
> ['2/7 추가: 최신 버전 역시 여전히 느립니다. 피드백 반영도 안 하시면서 의견은 왜 올려달라는 건지 모르겠네요. 앱 켜면 광고로 시작해서 첫 화면은 오늘드림 강조하느라 상단 메뉴 가리고, 올라이브 할 때에는 관련 배너 고정으로 띄워서 하단 메뉴 가리고... 사용자의 불편 사항에 대한 개선 의지가 아예 없으신 것 같습니다. 앱을 이용하면 할수록 사용자 편의성은 전혀 고려하지 않고, 한 화면에 최대한 많은 상품과 행사를 노출시키겠다는 집념만 느껴지네요. 이 점에만 올인하시니 이젠 화면을 터치하는 것도 밀려서 엉뚱한게 터치되기 일쑤입니다. 업데이트 될수록 앱 로딩도 더 오래 걸리고요. 심플함의 미학까진 바라지도 않으니 제발 UI/UX 개선에 조금이라도 관심을 가져 주세요. 앱 이용이 편리해야 뭔가를 주문하고 싶은 마음이 들텐데, 현재로선 쓰면 쓸수록 스트레스 받아서 앱을 아예 삭제하고 싶을 정도입니다. 가볍고 편리해서 다시 쓰고 싶어지는 앱을 만들어주세요. 부탁드립니다.',
>  '앱이 너무 느림, 리뷰달때 확인도 없이 바로 ...
>                                        ...                        
> ```



```python
from konlpy.tag import Mecab
mecab = Mecab()
```



```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer()
word_count_vector = count_vect.fit_transform(docs)
```



```python
count_vect.vocabulary_
```

> ```
> {'추가': 9800,
>  '최신': 9769,
>  '버전': 4580,
>  '역시': 7443,
>                                         ...                        
>  }
> ```



2. 리뷰 합쳐서 textrank(단어)



```python
doc_1 = (",").join(docs)
doc_1
```

> ```
> '2/7 추가: 최신 버전 역시 여전히 느립니다. 피드백 반영도 안 하시면서 의견은 왜 올려달라는 건지 모르겠네요. 앱 켜면 광고로 시작해서 첫 화면은 오늘드림 강조하느라 상단 메뉴 가리고, 올라이브 할 때에는 관련 배너 고정으로 띄워서 하단 메뉴 가리고... 사용자의 불편 사항에 대한 개선 의지가 아예 없으신 것 같습니다. 앱을 이용하면 할수록 사용자 편의성은 전혀 고려하지 않고, 한 화면에 최대한 많은 상품과 행사를 노출시키겠다는 집념만 느껴지네요. 이 점에만 올인하시니 이젠 화면을 터치하는 것도 밀려서 엉뚱한게 터치되기 일쑤입니다. 업데이트 될수록 앱 로딩도 더 오래 걸리고요. 심플함의 미학까진 바라지도 않으니 제발 UI/UX 개선에 조금이라도 관심을 가져 주세요. 앱 이용이 편리해야 뭔가를 주문하고 싶은 마음이 들텐데, 현재로선 쓰면 쓸수록 스트레스 받아서 앱을 아예 삭제하고 싶을 정도입니다. 가볍고 편리해서 다시 쓰고 싶어지는 앱을 만들어주세요. 부탁드립니다.,앱이 너무 느림, 리뷰달때 확인도 없이 바로 ...
>                                         ...                        
> ```



```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer()
word_count_vector = count_vect.fit_transform(words)
```



```python
count_vect.vocabulary_
```

> ```
> {'추가': 1427,
>  '최신': 1419,
>  '버전': 613,
>  '립니': 437,
>                                         ...                        
>  }
> ```



```python
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(word_count_vector)
```

> ```
> TfidfTransformer()
> ```



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

> ```
> [('로그인', 0.4582687367569976),
>  ('결제', 0.22683151040484553),
>  ('쿠폰', 0.213014362562926),
>  ('사용', 0.19804578573417986),
>  ('리뷰', 0.1957429277605266),
>  ('올리브', 0.19344006978687334),
>  ('배송', 0.18538006687908695),
>  ('구매', 0.1830772089054337),
>  ('화면', 0.17962292194495383),
>  ('업데이트', 0.17386577701082068)]
> ```



3. 리뷰 합쳐서 textrank



```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
BoW = count_vect.fit_transform(docs)
BoW.todense()
tfidf_trans = TfidfTransformer()
tfidf = tfidf_trans.fit_transform(BoW)
```



```python
import pandas as pd

vocab = count_vect.get_feature_names_out()
df = pd.DataFrame(tfidf.todense(), columns=vocab)
```



```python
terms = count_vect.get_feature_names()
# sum tfidf frequency of each term through documents
sums = tfidf.sum(axis=0)
# connecting term to its sums frequency
data = []
for col, term in enumerate(terms):
    data.append( (term, sums[0,col] ))
ranking = pd.DataFrame(data, columns=['term','rank'])
print(ranking.sort_values('rank', ascending=False))
```

> ```
>        term        rank
> 9162    좋아요  100.178625
> 2034     너무   48.355787
> 2291    느려요   24.920434
> 3266   로그인이   24.033411
> 1108     계속   23.884797
> ...     ...         ...
> 3075     때만    0.092768
> 4348   받는건데    0.092768
> 5678     소리    0.092768
> 7190  어플이라니    0.092768
> 4247    바꿔도    0.092768
> 
> [11240 rows x 2 columns]
> ```

